//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "CompilerInvocation.hpp"

#include <mlir/Bytecode/BytecodeReader.h>
#include <mlir/Bytecode/BytecodeWriter.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Analysis/GlobalsModRef.h>
#include <llvm/Bitcode/BitcodeWriterPass.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IRPrinter/IRPrintingPasses.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ThreadPool.h>
#include <llvm/Transforms/Instrumentation/AddressSanitizer.h>
#include <llvm/Transforms/Instrumentation/ThreadSanitizer.h>
#include <llvm/Transforms/Scalar/DeadStoreElimination.h>

#include <pylir/CodeGen/CodeGen.hpp>
#include <pylir/LLVM/PlaceStatepoints.hpp>
#include <pylir/LLVM/PylirGC.hpp>
#include <pylir/Optimizer/ExternalModels/ExternalModels.hpp>
#include <pylir/Optimizer/Linker/Linker.hpp>
#include <pylir/Optimizer/Optimizer.hpp>
#include <pylir/Optimizer/PylirHIR/IR/PylirHIRDialect.hpp>
#include <pylir/Optimizer/PylirMem/IR/PylirMemDialect.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/Transforms/Passes.hpp>
#include <pylir/Parser/Dumper.hpp>
#include <pylir/Parser/Parser.hpp>

#include "DiagnosticMessages.hpp"

using namespace mlir;
using namespace pylir::cli;

namespace {
bool enableLTO(const pylir::cli::CommandLine& commandLine) {
  const auto& args = commandLine.getArgs();
  if (args.hasFlag(OPT_flto, OPT_fno_lto, false))
    return true;

  // --lld-path overrides -f[no-]integrated-lld unconditionally. If the embedded
  // lld is used and -O4 enable LTO
  bool enable = !args.hasArg(OPT_lld_path_EQ) &&
                args.getLastArgValue(OPT_O, "0") == "4" &&
                args.hasFlag(OPT_fintegrated_lld, OPT_fno_integrated_lld, true);
  if (commandLine.verbose()) {
    if (enable)
      llvm::errs() << "Enabling LTO as integrated LLD and -O4 was enabled\n";
    else if (args.getLastArgValue(OPT_O, "0") == "4" &&
             (args.hasArg(OPT_lld_path_EQ) ||
              !args.hasFlag(OPT_fintegrated_lld, OPT_fno_integrated_lld, true)))
      llvm::errs() << "LTO not enabled as integrated LLD is not used. Add "
                      "'-flto' if your linker supports LTO\n";
  }
  return enable;
}

// https://cmake.org/cmake/help/latest/command/add_custom_command.html for
// format
llvm::SmallString<100> escapeForMakefile(llvm::StringRef filename) {
  llvm::SmallString<100> result;
  for (char c : filename) {
    switch (c) {
    case '$':
      // Single $ turn to two $.
      result.push_back('$');
      break;
    case ' ':
    case '#':
      // These just get escaped.
      result.push_back('\\');
      break;
    default: break;
    }
    result.push_back(c);
  }
  return result;
}

void writeDependencyOutput(llvm::raw_ostream& ostream,
                           llvm::StringRef outputFilename,
                           llvm::ArrayRef<std::string> additionalInputFiles) {
  ostream << escapeForMakefile(outputFilename) << ": ";
  llvm::interleave(llvm::map_range(additionalInputFiles, escapeForMakefile),
                   ostream, " ");
  ostream << '\n';
}

void llvmDataLayoutToMLIRDataLayout(mlir::ModuleOp moduleOp,
                                    const llvm::DataLayout& dataLayout) {
  moduleOp.getContext()->loadDialect<mlir::DLTIDialect>();
  moduleOp.getContext()->loadDialect<mlir::LLVM::LLVMDialect>();
  llvm::SmallVector<mlir::DataLayoutEntryInterface> entries;
  // Endian.
  entries.emplace_back(mlir::DataLayoutEntryAttr::get(
      mlir::StringAttr::get(moduleOp.getContext(),
                            mlir::DLTIDialect::kDataLayoutEndiannessKey),
      mlir::StringAttr::get(
          moduleOp.getContext(),
          dataLayout.isBigEndian()
              ? mlir::DLTIDialect::kDataLayoutEndiannessBig
              : mlir::DLTIDialect::kDataLayoutEndiannessLittle)));
  // Index size.
  auto i64 = mlir::IntegerType::get(moduleOp.getContext(), 64);
  entries.emplace_back(mlir::DataLayoutEntryAttr::get(
      mlir::IndexType::get(moduleOp.getContext()),
      mlir::IntegerAttr::get(i64, dataLayout.getIndexSizeInBits(0))));

  // Only integer types we actually care about (probably even more than what we
  // care about to be honest. We only really use whatever is index).
  llvm::LLVMContext context;
  for (std::size_t integerSize : {1, 8, 16, 32, 64}) {
    auto* llvmIntegerType = llvm::IntegerType::get(context, integerSize);
    auto mlirIntegerType =
        mlir::IntegerType::get(moduleOp.getContext(), integerSize);
    auto abiAlign = dataLayout.getABITypeAlign(llvmIntegerType);
    auto prefAlign = dataLayout.getPrefTypeAlign(llvmIntegerType);
    entries.emplace_back(mlir::DataLayoutEntryAttr::get(
        mlirIntegerType, mlir::DenseIntElementsAttr::get<std::uint64_t>(
                             mlir::VectorType::get({2}, i64),
                             {abiAlign.value() * 8, prefAlign.value() * 8})));
  }

  // Only really care about Double at the moment.
  auto* llvm64 =
      llvm::Type::getFloatingPointTy(context, llvm::APFloat::IEEEdouble());
  entries.emplace_back(mlir::DataLayoutEntryAttr::get(
      mlir::FloatType::getF64(moduleOp->getContext()),
      mlir::DenseIntElementsAttr::get<std::uint64_t>(
          mlir::VectorType::get({2}, i64),
          {dataLayout.getABITypeAlign(llvm64).value() * 8,
           dataLayout.getPrefTypeAlign(llvm64).value() * 8})));

  // Pointers
  entries.emplace_back(mlir::DataLayoutEntryAttr::get(
      mlir::LLVM::LLVMPointerType::get(moduleOp.getContext()),
      mlir::DenseIntElementsAttr::get<std::uint64_t>(
          mlir::VectorType::get({3}, i64),
          {dataLayout.getPointerSizeInBits(0),
           dataLayout.getPointerABIAlignment(0).value() * 8,
           dataLayout.getPointerPrefAlignment(0).value() * 8})));

  moduleOp->setAttr(
      mlir::DLTIDialect::kDataLayoutAttrName,
      mlir::DataLayoutSpecAttr::get(moduleOp.getContext(), entries));
}
} // namespace

mlir::LogicalResult pylir::CompilerInvocation::executeAction(
    llvm::opt::Arg* inputFile, CommandLine& commandLine,
    const pylir::Toolchain& toolchain, CompilerInvocation::Action action,
    Diag::DiagnosticsManager& diagManager) {
  const auto& args = commandLine.getArgs();

#if LLVM_ENABLE_THREADS
  if (args.hasFlag(OPT_Xmulti_threaded, OPT_Xsingle_threaded, true))
    m_threadPool = std::make_unique<llvm::StdThreadPool>();
  else
#endif
    m_threadPool = std::make_unique<llvm::SingleThreadExecutor>(
        llvm::hardware_concurrency(1));

  std::optional<llvm::ToolOutputFile> outputFile;
  if (!commandLine.onlyPrint()) {
    if (auto* arg = args.getLastArg(OPT_M)) {
      std::error_code ec;
      outputFile.emplace(arg->getValue(), ec, llvm::sys::fs::OF_Text);
      if (ec) {
        commandLine
            .createError(arg, Diag::FAILED_TO_OPEN_FILE_N, arg->getValue())
            .addHighlight(arg);
        return mlir::failure();
      }
      if (arg = args.getLastArg(OPT_o);
          arg && arg->getValue() == llvm::StringRef{"-"}) {
        commandLine
            .createError(
                arg, Diag::OUTPUT_CANNOT_BE_STDOUT_WHEN_WRITING_DEPENDENCY_FILE)
            .addHighlight(arg);
        return mlir::failure();
      }
    }
    if (mlir::failed(compilation(inputFile, commandLine, toolchain, action,
                                 diagManager)))
      return mlir::failure();
  }
  if (outputFile) {
    // The very first document is the main input file which build systems
    // already depend on. Hence, there is no need to add it to the output.
    std::vector<std::string> additionalInputFiles;
    if (!m_documents.empty()) {
      additionalInputFiles.resize(m_documents.size() - 1);
      llvm::transform(llvm::drop_begin(m_documents),
                      additionalInputFiles.begin(),
                      [](const Diag::Document& document) {
                        return document.getFilename();
                      });
    }

    // The document order is non-deterministic when multi threading is enabled.
    // Sort them to make it deterministic again.
    std::sort(additionalInputFiles.begin(), additionalInputFiles.end());
    writeDependencyOutput(outputFile->os(), m_actionOutputFilename,
                          additionalInputFiles);
    outputFile->keep();
  }
  if (action != Link)
    return finalizeOutputStream(mlir::success(), commandLine);

  if (commandLine.onlyPrint())
    if (mlir::failed(ensureOutputStream(args, action, commandLine)))
      return mlir::failure();

  PYLIR_ASSERT(m_tempFile);
  auto fileName = m_tempFile->TmpName;
  m_outFileStream.reset();
  if (auto error = m_tempFile->keep()) {
    llvm::consumeError(std::move(error));
    commandLine.createError(pylir::Diag::FAILED_TO_KEEP_TEMPORARY_FILE_N,
                            m_tempFile->TmpName);
    return mlir::failure();
  }
  bool success = toolchain.link(commandLine, fileName);
  llvm::sys::fs::remove(fileName);
  return mlir::success(success);
}

namespace {
mlir::FailureOr<std::string> readWholeFile(llvm::opt::Arg* inputFile,
                                           CommandLine& commandLine) {
  auto fd = llvm::sys::fs::openNativeFileForRead(inputFile->getValue());
  if (!fd) {
    llvm::consumeError(fd.takeError());
    commandLine
        .createError(inputFile, pylir::Diag::FAILED_TO_OPEN_FILE_N,
                     inputFile->getValue())
        .addHighlight(inputFile);
    return mlir::failure();
  }
  std::optional exit =
      llvm::make_scope_exit([&fd] { llvm::sys::fs::closeFile(*fd); });
  llvm::sys::fs::file_status status;
  {
    auto error = llvm::sys::fs::status(*fd, status);
    if (error) {
      commandLine
          .createError(inputFile, pylir::Diag::FAILED_TO_ACCESS_FILE_N,
                       inputFile->getValue())
          .addHighlight(inputFile);
      return mlir::failure();
    }
  }
  std::string content(status.getSize(), '\0');
  auto read =
      llvm::sys::fs::readNativeFile(*fd, {content.data(), content.size()});
  if (!read) {
    llvm::consumeError(fd.takeError());
    commandLine
        .createError(inputFile, pylir::Diag::FAILED_TO_READ_FILE_N,
                     inputFile->getValue())
        .addHighlight(inputFile);
    return mlir::failure();
  }
  return {std::move(content)};
}

mlir::ModuleOp buildModuleIfNecessary(std::unique_ptr<mlir::Block>&& block,
                                      mlir::MLIRContext* context) {
  if (llvm::hasSingleElement(*block))
    if (auto module = llvm::dyn_cast<mlir::ModuleOp>(&block->front())) {
      module->remove();
      return module;
    }

  mlir::OpBuilder builder(context);
  auto mlirModule = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
  mlir::Block& moduleBody = mlirModule.getBodyRegion().front();
  moduleBody.getOperations().splice(moduleBody.begin(), block->getOperations());
  return mlirModule;
}

} // namespace

mlir::LogicalResult pylir::CompilerInvocation::compilation(
    llvm::opt::Arg* inputFile, CommandLine& commandLine,
    const pylir::Toolchain& toolchain, CompilerInvocation::Action action,
    Diag::DiagnosticsManager& diagManager) {
  auto inputExtension = llvm::sys::path::extension(inputFile->getValue());
  auto type = llvm::StringSwitch<FileType>(inputExtension)
                  .Case(".py", FileType::Python)
                  .Cases(".mlir", ".mlirbc", FileType::MLIR)
                  .Cases(".ll", ".bc", FileType::LLVM)
                  .Default(FileType::Python);
  const auto& args = commandLine.getArgs();

  auto shouldOutput = [&args](pylir::cli::ID id) {
    auto* outputFormatArg = args.getLastArg(OPT_emit_llvm, OPT_emit_pylir);
    return outputFormatArg && outputFormatArg->getOption().getID() == id;
  };

  mlir::OwningOpRef<mlir::ModuleOp> mlirModule;
  std::unique_ptr<llvm::Module> llvmModule;
  switch (type) {
  case FileType::Python: {
    auto content = readWholeFile(inputFile, commandLine);
    if (mlir::failed(content))
      return mlir::failure();

    auto& document = addDocument(std::move(*content), inputFile->getValue());
    auto subDiagManager = diagManager.createSubDiagnosticManager(document);
    Syntax::FileInput* fileInput;
    {
      pylir::Parser parser(subDiagManager);
      auto tree = parser.parseFileInput();
      if (!tree || subDiagManager.errorsOccurred())
        return mlir::failure();

      fileInput = &m_fileInputs.emplace_back(std::move(*tree));
    }
    if (args.hasArg(OPT_dump_ast)) {
      pylir::Dumper dumper;
      llvm::outs() << dumper.dump(*fileInput);
    }
    if (action == SyntaxOnly)
      return mlir::success();

    ensureMLIRContext();

    auto module =
        codegenPythonToMLIR(args, commandLine, diagManager, subDiagManager);
    if (mlir::failed(module))
      return mlir::failure();

    mlirModule = std::move(*module);
    if (mlir::failed(ensureTargetMachine(args, commandLine, toolchain)))
      return mlir::failure();

    llvmDataLayoutToMLIRDataLayout(*mlirModule,
                                   m_targetMachine->createDataLayout());
    [[fallthrough]];
  }
  case FileType::MLIR: {
    if (type == FileType::MLIR) {
      ensureMLIRContext();
      m_mlirContext->allowUnregisteredDialects(false);
      FailureOr<std::string> content = readWholeFile(inputFile, commandLine);
      if (failed(content))
        return failure();

      auto config = ParserConfig(&*m_mlirContext, /*verifyAfterParse=*/true);
      if (auto buffer = llvm::MemoryBufferRef(*content, inputFile->getValue());
          isBytecode(buffer)) {
        std::unique_ptr<Block> body = std::make_unique<mlir::Block>();
        if (failed(readBytecodeFile(buffer, body.get(), config)))
          return failure();

        mlirModule = buildModuleIfNecessary(std::move(body), &*m_mlirContext);
      } else {
        mlirModule = parseSourceString<ModuleOp>(*content, config,
                                                 inputFile->getValue());
        // Call above should have emitted an error message through the context's
        // diagnostic handler in the error case.
        if (!mlirModule)
          return failure();
      }
    }
    mlir::PassManager manager(&*m_mlirContext);
    if (args.hasArg(OPT_Xstatistics))
      manager.enableStatistics();

#ifndef NDEBUG
    manager.enableVerifier();
#if !defined(__MINGW32_MAJOR_VERSION) || !defined(__clang__)
    manager.enableCrashReproducerGeneration("failure.mlir");
#endif
#else
    manager.enableVerifier(false);
#endif
    if (args.hasArg(OPT_Xprint_before, OPT_Xprint_after,
                    OPT_Xprint_after_all)) {
      bool afterAll = args.hasArg(OPT_Xprint_after_all);
      auto afterName = args.getLastArgValue(OPT_Xprint_after);
      auto beforeName = args.getLastArgValue(OPT_Xprint_before);
      manager.enableIRPrinting(
          [beforeName](mlir::Pass* pass, mlir::Operation*) {
            return pass->getName().equals_insensitive(beforeName);
          },
          [afterName, afterAll](mlir::Pass* pass, mlir::Operation*) {
            return afterAll || pass->getName().equals_insensitive(afterName);
          },
          args.hasArg(OPT_Xprint_module_scope), false, false, llvm::errs(),
          mlir::OpPrintingFlags().enableDebugInfo().assumeVerified());
    }
    if (args.hasArg(OPT_Xtiming))
      manager.enableTiming();

    bool produceDebugInfo =
        args.getLastArgValue(OPT_g, "0") != llvm::StringRef{"0"};
    if (!produceDebugInfo)
      manager.addPass(mlir::createStripDebugInfoPass());

    if (!shouldOutput(OPT_emit_pylir))
      if (mlir::failed(mlir::parsePassPipeline(
              args.getLastArgValue(OPT_O, "0") == "0" ? "pylir-minimum"
                                                      : "pylir-optimize",
              manager)))
        return mlir::failure();

    if (shouldOutput(OPT_emit_pylir)) {
      if (mlir::failed(manager.run(*mlirModule)))
        return mlir::failure();

      if (mlir::failed(ensureOutputStream(args, action, commandLine)))
        return mlir::failure();

      if (action == Assembly)
        mlirModule->print(
            *m_output,
            mlir::OpPrintingFlags{}.assumeVerified().enableDebugInfo());
      else
        (void)mlir::writeBytecodeToFile(
            *mlirModule, *m_output,
            mlir::BytecodeWriterConfig("Pylir " PYLIR_VERSION));

      return finalizeOutputStream(mlir::success(), commandLine);
    }
    if (mlir::failed(ensureTargetMachine(args, commandLine, toolchain)))
      return mlir::failure();

    auto options = pylir::PylirLLVMOptions(
        m_targetMachine->getTargetTriple().str(),
        m_targetMachine->createDataLayout().getStringRepresentation(),
        produceDebugInfo);
    if (mlir::failed(mlir::parsePassPipeline("pylir-llvm" + options.rendered(),
                                             manager)))
      return mlir::failure();

    if (args.hasArg(OPT_Xprint_pipeline)) {
      manager.printAsTextualPipeline(llvm::errs());
      llvm::errs() << '\n';
    }
    if (mlir::failed(manager.run(*mlirModule)))
      return mlir::failure();

    mlir::registerLLVMDialectTranslation(*m_mlirContext);
    mlir::registerBuiltinDialectTranslation(*m_mlirContext);
    llvmModule = mlir::translateModuleToLLVMIR(*mlirModule, *m_llvmContext);
    // Delete these now to release MLIRs resource and reduce peak memory usage.
    mlirModule = nullptr;
    m_mlirContext.reset();
    m_fileInputs.clear();
    m_documents.clear();

    [[fallthrough]];
  }
  case FileType::LLVM: {
    pylir::linkInGCStrategy();
    if (type == LLVM) {
      if (mlir::failed(ensureLLVMInit(args, toolchain)))
        return mlir::failure();

      llvm::SMDiagnostic diag;
      llvmModule =
          llvm::parseIRFile(inputFile->getValue(), diag, *m_llvmContext);
      if (!llvmModule) {
        std::string progName =
            llvm::sys::path::filename(commandLine.getExecutablePath()).str();
        diag.print(progName.c_str(), llvm::errs());
        return mlir::failure();
      }

      auto triple = llvmModule->getTargetTriple();
      if (mlir::failed(ensureTargetMachine(args, commandLine, toolchain,
                                           triple.empty()
                                               ? std::optional<llvm::Triple>{}
                                               : llvm::Triple(triple))))
        return mlir::failure();

      if (llvmModule->getDataLayout().isDefault())
        llvmModule->setDataLayout(m_targetMachine->createDataLayout());

      if (triple.empty()) {
        // Clang warns about this case by default, which I personally think is
        // odd. Something to consider in the future if convinced otherwise
        // however.
        llvmModule->setTargetTriple(m_targetMachine->getTargetTriple().str());
      }
    }

    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;

    llvm::PassInstrumentationCallbacks pic;
    llvm::StandardInstrumentations si(*m_llvmContext, false);
    si.registerCallbacks(pic, &mam);

    llvm::PipelineTuningOptions options;
    options.LoopInterleaving = true;
    options.LoopUnrolling = true;
    options.LoopVectorization = true;
    options.SLPVectorization = true;
    options.MergeFunctions = true;
    llvm::PassBuilder passBuilder(m_targetMachine.get(), options, std::nullopt,
                                  &pic);

    passBuilder.registerOptimizerLastEPCallback(
        [&](llvm::ModulePassManager& mpm, llvm::OptimizationLevel) {
          mpm.addPass(pylir::PlaceStatepointsPass{});
          if (args.getLastArgValue(OPT_O, "0") != "0") {
            // Cleanup redundant stores to allocas inserted when statepoints
            // were placed.
            mpm.addPass(
                llvm::createModuleToFunctionPassAdaptor(llvm::DSEPass{}));
          }
        });

    fam.registerPass([&] { return passBuilder.buildDefaultAAPipeline(); });
    passBuilder.registerModuleAnalyses(mam);
    passBuilder.registerCGSCCAnalyses(cgam);
    passBuilder.registerFunctionAnalyses(fam);
    passBuilder.registerLoopAnalyses(lam);
    passBuilder.crossRegisterProxies(lam, fam, cgam, mam);

    bool lto = enableLTO(commandLine);
    llvm::ModulePassManager mpm;
    if (args.getLastArgValue(OPT_O, "0") == "0") {
      mpm =
          passBuilder.buildO0DefaultPipeline(llvm::OptimizationLevel::O0, lto);
    } else {
      llvm::OptimizationLevel level =
          llvm::StringSwitch<llvm::OptimizationLevel>(
              args.getLastArgValue(OPT_O))
              .Case("1", llvm::OptimizationLevel::O1)
              .Case("2", llvm::OptimizationLevel::O2)
              .Case("3", llvm::OptimizationLevel::O3)
              .Case("4", llvm::OptimizationLevel::O3)
              .Case("s", llvm::OptimizationLevel::Os)
              .Case("z", llvm::OptimizationLevel::Oz);
      if (lto)
        mpm = passBuilder.buildLTOPreLinkDefaultPipeline(level);
      else
        mpm = passBuilder.buildPerModuleDefaultPipeline(level);
    }

    if (shouldOutput(OPT_emit_llvm) || lto) {
      // See
      // https://github.com/llvm/llvm-project/blob/ea22fdd120aeb1bbb9ea96670d70193dc02b2c5f/clang/lib/CodeGen/BackendUtil.cpp#L1467
      // Doing full LTO for now
      bool emitLTOSummary =
          lto &&
          m_targetMachine->getTargetTriple().getVendor() != llvm::Triple::Apple;
      if (emitLTOSummary) {
        if (!llvmModule->getModuleFlag("ThinLTO"))
          llvmModule->addModuleFlag(llvm::Module::Error, "ThinLTO",
                                    std::uint32_t(0));
        if (!llvmModule->getModuleFlag("EnableSplitLTOUnit"))
          llvmModule->addModuleFlag(llvm::Module::Error, "EnableSplitLTOUnit",
                                    std::uint32_t(1));
      }
      if (mlir::failed(ensureOutputStream(args, action, commandLine)))
        return mlir::failure();

      if (action == pylir::CompilerInvocation::Assembly)
        mpm.addPass(llvm::PrintModulePass(*m_output));
      else
        mpm.addPass(llvm::BitcodeWriterPass(*m_output, false, emitLTOSummary));
    }

    mpm.run(*llvmModule, mam);
    if (args.hasArg(OPT_Xprint_pipeline))
      mpm.printPipeline(llvm::errs(), [&](llvm::StringRef className) {
        auto passName = pic.getPassNameForClassName(className);
        return passName.empty() ? className : passName;
      });

    if (shouldOutput(OPT_emit_llvm))
      return finalizeOutputStream(mlir::success(), commandLine);

    if (lto)
      break;

    if (mlir::failed(ensureOutputStream(args, action, commandLine)))
      return mlir::failure();

    llvm::legacy::PassManager codeGenPasses;
    codeGenPasses.add(llvm::createTargetTransformInfoWrapperPass(
        m_targetMachine->getTargetIRAnalysis()));
    if (m_targetMachine->addPassesToEmitFile(
            codeGenPasses, *m_output, nullptr,
            action == pylir::CompilerInvocation::Assembly
                ? llvm::CodeGenFileType::AssemblyFile
                : llvm::CodeGenFileType::ObjectFile)) {
      std::string_view format = action == pylir::CompilerInvocation::Assembly
                                    ? "Assembly"
                                    : "Object file";
      auto* arg = args.getLastArg(OPT_target_EQ);
      if (!arg)
        arg = args.getLastArg(OPT_c, OPT_S);

      if (arg) {
        commandLine
            .createError(arg,
                         pylir::Diag::TARGET_N_DOES_NOT_SUPPORT_COMPILING_TO_N,
                         m_targetMachine->getTargetTriple().str(), format)
            .addHighlight(arg);
        return finalizeOutputStream(mlir::failure(), commandLine);
      }
      commandLine.createError(
          pylir::Diag::TARGET_N_DOES_NOT_SUPPORT_COMPILING_TO_N,
          m_targetMachine->getTargetTriple().str(), format);
      return finalizeOutputStream(mlir::failure(), commandLine);
    }

    codeGenPasses.run(*llvmModule);
    break;
  }
  }
  return mlir::success();
}

void pylir::CompilerInvocation::ensureMLIRContext() {
  if (m_mlirContext)
    return;

  mlir::DialectRegistry registry;
  registry.insert<pylir::Py::PylirPyDialect>();
  registry.insert<pylir::HIR::PylirHIRDialect>();
  registry.insert<pylir::Mem::PylirMemDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::DLTIDialect>();
  pylir::registerExternalModels(registry);
  m_mlirContext.emplace(registry);
  m_mlirContext->disableMultithreading();
  m_mlirContext->setThreadPool(*m_threadPool);
  m_mlirContext->getDiagEngine().registerHandler(
      [](mlir::Diagnostic& diagnostic) {
        diagnostic.print(llvm::errs());
        llvm::errs() << '\n';
      });
}

namespace {
std::string formDefaultOutputName(const llvm::opt::InputArgList& args,
                                  pylir::CompilerInvocation::Action action) {
  std::string inputFilename =
      llvm::sys::path::filename(args.getLastArgValue(OPT_INPUT)).str();

  auto extension = llvm::sys::path::extension(inputFilename);
  bool isKnownFileExtension = llvm::is_contained<llvm::StringRef>(
      {".py", ".mlir", ".ll", ".bc", ".mlirbc"}, extension);
  if (isKnownFileExtension)
    inputFilename.resize(inputFilename.size() - extension.size());

  std::string defaultName;
  if (args.hasArg(OPT_emit_pylir)) {
    if (action == pylir::CompilerInvocation::ObjectFile)
      defaultName = inputFilename + ".mlirbc";
    else
      defaultName = inputFilename + ".mlir";
  } else if (args.hasArg(OPT_emit_llvm)) {
    if (action == pylir::CompilerInvocation::ObjectFile)
      defaultName = inputFilename + ".bc";
    else
      defaultName = inputFilename + ".ll";
  } else if (action == pylir::CompilerInvocation::Assembly) {
    defaultName = inputFilename + ".s";
  } else if (action == pylir::CompilerInvocation::ObjectFile) {
    defaultName = inputFilename + ".o";
  }
  return defaultName;
}
} // namespace

mlir::LogicalResult pylir::CompilerInvocation::ensureOutputStream(
    const llvm::opt::InputArgList& args, Action action,
    cli::CommandLine& commandLine) {
  if (m_output)
    return mlir::success();

  llvm::SmallString<20> compilerOutputFilepath;
  if (action == pylir::CompilerInvocation::Link) {
    llvm::sys::path::system_temp_directory(true, compilerOutputFilepath);
    llvm::sys::path::append(compilerOutputFilepath, "tmp.o");
  } else {
    compilerOutputFilepath =
        args.getLastArgValue(OPT_o, formDefaultOutputName(args, action));
  }

  bool useTemporaryFile = true;
  if (compilerOutputFilepath == "-") {
    useTemporaryFile = false;
  } else {
    llvm::sys::fs::file_status status;
    llvm::sys::fs::status(compilerOutputFilepath, status);
    if (llvm::sys::fs::exists(status) &&
        !llvm::sys::fs::is_regular_file(status)) {
      useTemporaryFile = false;
    }
  }

  if (useTemporaryFile) {
    llvm::SmallString<20> tempFileName = compilerOutputFilepath;
    auto extension = llvm::sys::path::extension(compilerOutputFilepath);
    llvm::sys::path::remove_filename(tempFileName);
    llvm::sys::path::append(tempFileName,
                            llvm::sys::path::stem(compilerOutputFilepath) +
                                "-%%%%%%%%%" + extension + ".tmp");
    auto tempFile = llvm::sys::fs::TempFile::create(tempFileName);
    if (!tempFile) {
      llvm::consumeError(tempFile.takeError());
      commandLine.createError(pylir::Diag::FAILED_TO_CREATE_TEMPORARY_FILE_N,
                              tempFileName.str());
      return mlir::failure();
    }
    m_tempFile = std::move(*tempFile);
    m_outFileStream.emplace(m_tempFile->FD, false);
  } else {
    std::error_code ec;
    m_outFileStream.emplace(compilerOutputFilepath, ec,
                            llvm::sys::fs::FileAccess::FA_Write);
    if (ec) {
      commandLine.createError(Diag::FAILED_TO_OPEN_OUTPUT_FILE_N_FOR_WRITING,
                              compilerOutputFilepath.str());
      return mlir::failure();
    }
  }

  m_output = &*m_outFileStream;
  m_compileStepOutputFilename = compilerOutputFilepath.str();
  m_actionOutputFilename = action == Link ? args.getLastArgValue(OPT_o)
                                          : m_compileStepOutputFilename;
  return mlir::success();
}

mlir::LogicalResult
pylir::CompilerInvocation::finalizeOutputStream(mlir::LogicalResult result,
                                                cli::CommandLine& commandLine) {
  m_outFileStream.reset();
  m_output = nullptr;
  if (!m_tempFile)
    return result;

  auto exit = llvm::make_scope_exit([&] { m_tempFile.reset(); });
  if (mlir::failed(result)) {
    if (auto error = m_tempFile->discard()) {
      llvm::consumeError(std::move(error));
      commandLine.createError(pylir::Diag::FAILED_TO_DISCARD_TEMPORARY_FILE_N,
                              m_tempFile->TmpName);
    }
    return mlir::failure();
  }
  if (auto error = m_tempFile->keep(m_compileStepOutputFilename)) {
    llvm::consumeError(std::move(error));
    commandLine.createError(pylir::Diag::FAILED_TO_RENAME_TEMPORARY_FILE_N_TO_N,
                            m_tempFile->TmpName, m_compileStepOutputFilename);
    return mlir::failure();
  }
  return result;
}

mlir::LogicalResult pylir::CompilerInvocation::ensureTargetMachine(
    const llvm::opt::InputArgList& args, CommandLine& commandLine,
    const pylir::Toolchain& toolchain, std::optional<llvm::Triple> triple) {
  if (mlir::failed(ensureLLVMInit(args, toolchain)))
    return mlir::failure();

  if (m_targetMachine)
    return mlir::success();

  if (!triple)
    triple = llvm::Triple(
        args.getLastArgValue(OPT_target_EQ, LLVM_DEFAULT_TARGET_TRIPLE));

  std::string error;
  const auto* targetM =
      llvm::TargetRegistry::lookupTarget(triple->str(), error);
  if (!targetM) {
    auto* outputArg = args.getLastArg(OPT_target_EQ);
    if (!outputArg) {
      commandLine.createError(pylir::Diag::COULD_NOT_FIND_TARGET_N,
                              triple->str());
      return mlir::failure();
    }
    commandLine
        .createError(outputArg, pylir::Diag::COULD_NOT_FIND_TARGET_N,
                     triple->str())
        .addHighlight(outputArg);
    return mlir::failure();
  }

  auto optLevel = llvm::StringSwitch<std::optional<llvm::CodeGenOptLevel>>(
                      args.getLastArgValue(OPT_O, "0"))
                      .Case("0", llvm::CodeGenOptLevel::None)
                      .Case("1", llvm::CodeGenOptLevel::Less)
                      .Case("2", llvm::CodeGenOptLevel::Default)
                      .Case("3", llvm::CodeGenOptLevel::Aggressive)
                      .Case("4", llvm::CodeGenOptLevel::Aggressive)
                      .Case("s", llvm::CodeGenOptLevel::Default)
                      .Case("z", llvm::CodeGenOptLevel::Default)
                      .Default(std::nullopt);
  if (!optLevel) {
    auto* optArg = args.getLastArg(OPT_O);
    commandLine
        .createError(optArg, pylir::Diag::INVALID_OPTIMIZATION_LEVEL_N,
                     optArg->getAsString(args))
        .addHighlight(optArg);
    return mlir::failure();
  }

  llvm::Reloc::Model relocation = llvm::Reloc::Static;
  if (toolchain.defaultsToPIC() || toolchain.isPIE(commandLine))
    relocation = llvm::Reloc::PIC_;

  llvm::TargetOptions targetOptions;
  if (triple->isOSWindows())
    targetOptions.ExceptionModel = llvm::ExceptionHandling::WinEH;
  else
    targetOptions.ExceptionModel = llvm::ExceptionHandling::DwarfCFI;

  targetOptions.UseInitArray = true;
  m_targetMachine = std::unique_ptr<llvm::TargetMachine>(
      targetM->createTargetMachine(triple->str(), "generic", "", targetOptions,
                                   relocation, {}, *optLevel));
  return mlir::success();
}

mlir::LogicalResult
pylir::CompilerInvocation::ensureLLVMInit(const llvm::opt::InputArgList& args,
                                          const pylir::Toolchain& toolchain) {
  if (m_llvmContext)
    return mlir::success();

  m_llvmContext = std::make_unique<llvm::LLVMContext>();
  std::vector<const char*> refs;
  refs.push_back("pylir (LLVM option parsing)");

  auto options = toolchain.getLLVMOptions(args);
  std::transform(options.begin(), options.end(), std::back_inserter(refs),
                 [](const std::string& str) { return str.c_str(); });
  refs.push_back(nullptr);
  llvm::cl::ResetAllOptionOccurrences();
  return mlir::success(llvm::cl::ParseCommandLineOptions(
      refs.size() - 1, refs.data(), "", &llvm::errs()));
}

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
pylir::CompilerInvocation::codegenPythonToMLIR(
    const llvm::opt::InputArgList& args, const cli::CommandLine& commandLine,
    Diag::DiagnosticsManager& diagManager,
    Diag::DiagnosticsDocManager<>& mainModuleDiagManager) {
  pylir::CodeGenOptions options{};
  options.implicitBuiltinsImport =
      args.hasFlag(OPT_Xbuiltins, OPT_Xno_builtins, true);
  std::vector<std::string> importPaths;
  {
    llvm::SmallString<100> docPath{m_documents.front().getFilename()};
    llvm::sys::fs::make_absolute(docPath);
    llvm::sys::path::remove_filename(docPath);
    importPaths.emplace_back(docPath);
  }
  for (auto& iter : args.getAllArgValues(OPT_I)) {
    llvm::SmallString<100> path{iter};
    llvm::sys::fs::make_absolute(path);
    importPaths.emplace_back(path);
  }

  auto appendEnvVar = [&](llvm::StringRef value) {
    llvm::SmallVector<llvm::StringRef> output;
    value.split(output, llvm::sys::EnvPathSeparator);
    llvm::transform(output, std::back_inserter(importPaths),
                    [](llvm::SmallString<100> path) {
                      llvm::sys::fs::make_absolute(path);
                      return std::string{path};
                    });
  };

  if (auto* pythonPath = std::getenv("PYTHONPATH"))
    appendEnvVar(pythonPath);

  if (auto* pythonHome = std::getenv("PYTHONHOME")) {
    appendEnvVar(pythonHome);
  } else {
    llvm::SmallString<100> libDirPath = commandLine.getExecutablePath();
    llvm::sys::path::remove_filename(libDirPath);
    llvm::sys::path::append(libDirPath, "..", "lib");
    importPaths.emplace_back(libDirPath);
  }

  // Protects 'futures' and 'loaded'.
  std::mutex dataStructureMutex;
  std::vector<std::pair<std::shared_future<mlir::ModuleOp>, std::string>>
      futures;
  llvm::StringMap<std::string> loaded;

  // Protects 'm_fileInputs' and 'm_documents'.
  std::mutex sourceDSMutex;

  llvm::ThreadPoolTaskGroup taskGroup(*m_threadPool);
  options.moduleLoadCallback =
      [&](llvm::StringRef absoluteModule,
          Diag::DiagnosticsDocManager<>* diagnostics,
          Diag::LazyLocation location) {
        std::unique_lock lock{dataStructureMutex};
        auto [iter, inserted] = loaded.try_emplace(absoluteModule, "");
        if (!inserted)
          return;

        // The path to check depends on whether this is a top level module (no
        // '.') or a submodule. Submodules can only be loaded after their parent
        // module was loaded and need to be in a subdirectory of its parent
        // package.
        std::vector<std::string> pathsToCheck;
        auto [parentPackage, thisModule] = absoluteModule.rsplit('.');
        if (thisModule.empty()) {
          // Top level module.
          pathsToCheck = importPaths;
          thisModule = parentPackage;
        } else {
          std::string parentPath = loaded.lookup(parentPackage);
          PYLIR_ASSERT(!parentPath.empty() &&
                       "parent package must have been loaded previously");
          if (!llvm::StringRef(parentPath).ends_with("__init__.py")) {
            // TODO: Should this be diagnosed? Probably.
          }
          pathsToCheck.emplace_back(llvm::sys::path::parent_path(parentPath));
        }

        llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer(
            std::errc::no_such_file_or_directory);
        for (llvm::StringRef candidate : pathsToCheck) {
          // Prefer packages to plain modules.
          llvm::SmallString<128> path = candidate;
          llvm::sys::path::append(path, thisModule, "__init__.py");

          buffer = llvm::MemoryBuffer::getFile(path);
          if (buffer)
            break;

          path = candidate;
          llvm::sys::path::append(path, thisModule + ".py");

          buffer = llvm::MemoryBuffer::getFile(path);
          if (buffer)
            break;
        }

        if (!buffer) {
          // TODO: This is a source of non-determinism as the location being
          // used
          //       is dependent on the scheduling of threads.
          Diag::DiagnosticsBuilder(*diagnostics, Diag::Severity::Error,
                                   location, Diag::FAILED_TO_FIND_MODULE_N,
                                   absoluteModule)
              .addHighlight(location);
          return;
        }
        iter->second = (*buffer)->getBufferIdentifier();

        auto action = [=, &sourceDSMutex, &diagManager, &options,
                       buffer = std::shared_ptr(std::move(*buffer)),
                       absoluteModule =
                           absoluteModule.str()]() mutable -> mlir::ModuleOp {
          std::unique_lock sourceLock{sourceDSMutex};
          Diag::Document& document = addDocument(
              buffer->getBuffer(), buffer->getBufferIdentifier().str());
          sourceLock.unlock();

          Diag::DiagnosticsDocManager docManager =
              diagManager.createSubDiagnosticManager(document);
          Parser parser(docManager);
          std::optional<Syntax::FileInput> tree = parser.parseFileInput();
          if (!tree || docManager.errorsOccurred())
            return nullptr;

          sourceLock.lock();
          Syntax::FileInput& fileInput =
              m_fileInputs.emplace_back(std::move(*tree));
          sourceLock.unlock();

          CodeGenOptions copyOption = options;
          copyOption.qualifier = std::move(absoluteModule);
          mlir::OwningOpRef<mlir::ModuleOp> res = pylir::codegenModule(
              &*m_mlirContext, fileInput, docManager, copyOption);
          if (docManager.errorsOccurred())
            return nullptr;

          return res.release();
        };

        futures.emplace_back(m_threadPool->async(taskGroup, std::move(action)),
                             absoluteModule.str());
      };

  // Also place the main module codegen a task on the threadpool. This is
  // purely for uniformity and debuggability when using '-Xsingle-threaded'.
  // Having all code stay in the same thread works better in 'gdb' and 'lldb'.
  {
    std::unique_lock lock{dataStructureMutex};

    options.qualifier = "__main__";
    futures.emplace_back(
        m_threadPool->async(taskGroup,
                            [&]() -> mlir::ModuleOp {
                              mlir::OwningOpRef<mlir::ModuleOp> mainModule =
                                  codegenModule(&*m_mlirContext,
                                                m_fileInputs.front(),
                                                mainModuleDiagManager, options);
                              if (mainModuleDiagManager.errorsOccurred())
                                return nullptr;

                              return mainModule.release();
                            }),
        options.qualifier);
  }

  // Wait for all codegen to be done.
  taskGroup.wait();

  // The contents of 'futures' is now final. Sort it into a deterministic order
  // for the linker step.
  std::sort(futures.begin(), futures.end(), llvm::less_second{});

  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> importedModules(
      futures.size());
  llvm::transform(llvm::make_first_range(futures), importedModules.begin(),
                  [](const std::shared_future<mlir::ModuleOp>& future) {
                    return future.get();
                  });

  // If any returned null, then there were errors.
  if (!llvm::all_of(importedModules,
                    llvm::identity<mlir::OwningOpRef<mlir::ModuleOp>>{}))
    return mlir::failure();

  return linkModules(importedModules);
}

pylir::Diag::Document&
pylir::CompilerInvocation::addDocument(std::string_view content,
                                       std::string filename) {
  auto& doc = m_documents.emplace_back(content, std::move(filename));
  if (m_verifier)
    m_verifier->addDocument(doc);

  return doc;
}
