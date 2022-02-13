
#include "CompilerInvocation.hpp"

#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Arithmetic/Transforms/Passes.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/Transforms/LegalizeForExport.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/ADT/ScopeExit.h>
#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Bitcode/BitcodeWriterPass.h>
#include <llvm/IR/IRPrintingPasses.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>

#include <pylir/CodeGen/CodeGen.hpp>
#include <pylir/Diagnostics/DiagnosticMessages.hpp>
#include <pylir/LLVM/PlaceStatepoints.hpp>
#include <pylir/LLVM/PylirGC.hpp>
#include <pylir/Optimizer/Conversion/Passes.hpp>
#include <pylir/Optimizer/PylirMem/IR/PylirMemDialect.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/Transforms/Passes.hpp>
#include <pylir/Optimizer/Transforms/Passes.hpp>
#include <pylir/Parser/Dumper.hpp>
#include <pylir/Parser/Parser.hpp>

using namespace pylir::cli;

namespace
{
bool enableLTO(const pylir::cli::CommandLine& commandLine)
{
    const auto& args = commandLine.getArgs();
    if (args.hasFlag(OPT_flto, OPT_fno_lto, false))
    {
        return true;
    }
#ifdef PYLIR_EMBEDDED_LLD
    // --ld-path overrides -f[no-]integrated-ld unconditionally. If the embedded ld
    // is used and -O4 enable LTO
    bool enable = !args.hasArg(OPT_ld_path_EQ) && args.getLastArgValue(OPT_O, "0") == "4"
                  && args.hasFlag(OPT_fintegrated_ld, OPT_fno_integrated_ld, true);
    if (commandLine.verbose())
    {
        if (enable)
        {
            llvm::errs() << "Enabling LTO as integrated LLD and -O4 was enabled\n";
        }
        else if (args.getLastArgValue(OPT_O, "0") == "4"
                 && (args.hasArg(OPT_ld_path_EQ) || !args.hasFlag(OPT_fintegrated_ld, OPT_fno_integrated_ld, true)))
        {
            llvm::errs() << "LTO not enabled as integrated LLD is not used. Add '-flto' if your linker supports LTO\n";
        }
    }
    return enable;
#else
    return false;
#endif
}
} // namespace

mlir::LogicalResult pylir::CompilerInvocation::executeAction(llvm::opt::Arg* inputFile,
                                                             const cli::CommandLine& commandLine,
                                                             const pylir::Toolchain& toolchain,
                                                             CompilerInvocation::Action action)
{
    auto inputExtension = llvm::sys::path::extension(inputFile->getValue());
    auto type = llvm::StringSwitch<FileType>(inputExtension)
                    .Case(".py", FileType::Python)
                    .Case(".mlir", FileType::MLIR)
                    .Cases(".ll", ".bc", FileType::LLVM)
                    .Default(FileType::Python);
    const auto& args = commandLine.getArgs();
    mlir::OwningOpRef<mlir::ModuleOp> mlirModule;
    std::unique_ptr<llvm::Module> llvmModule;
    switch (type)
    {
        case FileType::Python:
        {
            auto fd = llvm::sys::fs::openNativeFileForRead(inputFile->getValue());
            if (!fd)
            {
                llvm::consumeError(fd.takeError());
                llvm::errs() << commandLine
                                    .createDiagnosticsBuilder(inputFile, pylir::Diag::FAILED_TO_OPEN_FILE_N,
                                                              inputFile->getValue())
                                    .addLabel(inputFile, std::nullopt, pylir::Diag::ERROR_COLOUR)
                                    .emitError();
                return mlir::failure();
            }
            std::optional exit = llvm::make_scope_exit([&fd] { llvm::sys::fs::closeFile(*fd); });
            llvm::sys::fs::file_status status;
            {
                auto error = llvm::sys::fs::status(*fd, status);
                if (error)
                {
                    llvm::errs() << commandLine
                                        .createDiagnosticsBuilder(inputFile, pylir::Diag::FAILED_TO_ACCESS_FILE_N,
                                                                  inputFile->getValue())
                                        .addLabel(inputFile, std::nullopt, pylir::Diag::ERROR_COLOUR)
                                        .emitError();
                    return mlir::failure();
                }
            }
            std::string content(status.getSize(), '\0');
            auto read = llvm::sys::fs::readNativeFile(*fd, {content.data(), content.size()});
            if (!read)
            {
                llvm::consumeError(fd.takeError());
                llvm::errs() << commandLine
                                    .createDiagnosticsBuilder(inputFile, pylir::Diag::FAILED_TO_READ_FILE_N,
                                                              inputFile->getValue())
                                    .addLabel(inputFile, std::nullopt, pylir::Diag::ERROR_COLOUR)
                                    .emitError();
                return mlir::failure();
            }
            exit.reset();
            m_document = Diag::Document(std::move(content), inputFile->getValue());
            {
                pylir::Parser parser(*m_document);
                auto tree = parser.parseFileInput();
                if (!tree)
                {
                    llvm::errs() << tree.error();
                    return mlir::failure();
                }
                m_fileInput = std::move(*tree);
            }
            if (args.hasArg(OPT_emit_ast))
            {
                pylir::Dumper dumper;
                llvm::outs() << dumper.dump(*m_fileInput);
            }
            if (action == SyntaxOnly)
            {
                return mlir::success();
            }
            ensureMLIRContext(args);
            mlirModule = pylir::codegen(&*m_mlirContext, *m_fileInput, *m_document);
            [[fallthrough]];
        }
        case FileType::MLIR:
        {
            if (type == FileType::MLIR)
            {
                ensureMLIRContext(args);
                m_mlirContext->allowUnregisteredDialects(false);
                mlirModule = mlir::parseSourceFile<mlir::ModuleOp>(inputFile->getValue(), &*m_mlirContext);
                if (!mlirModule)
                {
                    return mlir::failure();
                }
            }
            mlir::PassManager manager(&*m_mlirContext);
            if (args.hasArg(OPT_Xstatistics))
            {
                manager.enableStatistics();
            }
#ifndef NDEBUG
            manager.enableVerifier();
    #if !defined(__MINGW32_MAJOR_VERSION) || !defined(__clang__)
            manager.enableCrashReproducerGeneration("failure.mlir");
    #endif
#endif
            if (args.hasArg(OPT_Xprint_before, OPT_Xprint_after, OPT_Xprint_after_all))
            {
                bool afterAll = args.hasArg(OPT_Xprint_after_all);
                auto afterName = args.getLastArgValue(OPT_Xprint_after);
                auto beforeName = args.getLastArgValue(OPT_Xprint_before);
                manager.enableIRPrinting([beforeName](mlir::Pass* pass, mlir::Operation*)
                                         { return pass->getName().equals_insensitive(beforeName); },
                                         [afterName, afterAll](mlir::Pass* pass, mlir::Operation*)
                                         { return afterAll || pass->getName().equals_insensitive(afterName); },
                                         false);
            }
            if (args.hasArg(OPT_Xtiming))
            {
                manager.enableTiming();
            }
            if (args.hasArg(OPT_emit_pylir))
            {
                if (mlir::failed(manager.run(*mlirModule)))
                {
                    return mlir::failure();
                }
                if (mlir::failed(ensureOutputStream(args, action)))
                {
                    return mlir::failure();
                }
                mlirModule->print(*m_output, mlir::OpPrintingFlags{}.enableDebugInfo());
                return finalizeOutputStream(mlir::success());
            }
            addOptimizationPasses(args.getLastArgValue(OPT_O, "0"), manager);
            if (args.hasArg(OPT_emit_mlir))
            {
                if (mlir::failed(manager.run(*mlirModule)))
                {
                    return mlir::failure();
                }
                if (mlir::failed(ensureOutputStream(args, action)))
                {
                    return mlir::failure();
                }
                mlirModule->print(*m_output, mlir::OpPrintingFlags{}.enableDebugInfo());
                return finalizeOutputStream(mlir::success());
            }
            if (mlir::failed(ensureTargetMachine(args, commandLine, toolchain)))
            {
                return mlir::failure();
            }
            manager.addNestedPass<mlir::FuncOp>(mlir::arith::createArithmeticExpandOpsPass());
            manager.addPass(pylir::Mem::createConvertPylirToLLVMPass(m_targetMachine->getTargetTriple(),
                                                                     m_targetMachine->createDataLayout()));
            manager.addNestedPass<mlir::LLVM::LLVMFuncOp>(mlir::createReconcileUnrealizedCastsPass());
            manager.addNestedPass<mlir::LLVM::LLVMFuncOp>(mlir::LLVM::createLegalizeForExportPass());
            if (mlir::failed(manager.run(*mlirModule)))
            {
                return mlir::failure();
            }
            mlir::registerLLVMDialectTranslation(*m_mlirContext);
            llvmModule = mlir::translateModuleToLLVMIR(*mlirModule, *m_llvmContext);
            [[fallthrough]];
        }
        case FileType::LLVM:
        {
            pylir::linkInGCStrategy();
            if (type == LLVM)
            {
                if (mlir::failed(ensureLLVMInit(args)))
                {
                    return mlir::failure();
                }
                llvm::SMDiagnostic diag;
                llvmModule = llvm::parseIRFile(inputFile->getValue(), diag, *m_llvmContext);
                if (diag.getSourceMgr())
                {
                    std::string progName = llvm::sys::path::filename(commandLine.getExecutablePath()).str();
                    diag.print(progName.c_str(), llvm::errs());
                    return mlir::failure();
                }

                auto triple = llvmModule->getTargetTriple();
                if (mlir::failed(
                        ensureTargetMachine(args, commandLine, toolchain,
                                            triple.empty() ? llvm::Optional<llvm::Triple>{} : llvm::Triple(triple))))
                {
                    return mlir::failure();
                }
            }

            llvm::LoopAnalysisManager lam;
            llvm::FunctionAnalysisManager fam;
            llvm::CGSCCAnalysisManager cgam;
            llvm::ModuleAnalysisManager mam;
            llvm::PassBuilder passBuilder(m_targetMachine.get());

            passBuilder.registerOptimizerLastEPCallback(
                [](llvm::ModulePassManager& mpm, llvm::OptimizationLevel)
                {
                    mpm.addPass(pylir::PlaceStatepointsPass{});
                });

            fam.registerPass([&] { return passBuilder.buildDefaultAAPipeline(); });
            passBuilder.registerModuleAnalyses(mam);
            passBuilder.registerCGSCCAnalyses(cgam);
            passBuilder.registerFunctionAnalyses(fam);
            passBuilder.registerLoopAnalyses(lam);
            passBuilder.crossRegisterProxies(lam, fam, cgam, mam);

            bool lto = enableLTO(commandLine);
            llvm::ModulePassManager mpm;
            if (args.getLastArgValue(OPT_O, "0") == "0")
            {
                mpm = passBuilder.buildO0DefaultPipeline(llvm::OptimizationLevel::O0, lto);
            }
            else
            {
                llvm::OptimizationLevel level = llvm::StringSwitch<llvm::OptimizationLevel>(args.getLastArgValue(OPT_O))
                                                    .Case("1", llvm::OptimizationLevel::O1)
                                                    .Case("2", llvm::OptimizationLevel::O2)
                                                    .Case("3", llvm::OptimizationLevel::O3)
                                                    .Case("4", llvm::OptimizationLevel::O3)
                                                    .Case("s", llvm::OptimizationLevel::Os)
                                                    .Case("z", llvm::OptimizationLevel::Oz);
                if (lto)
                {
                    mpm = passBuilder.buildLTOPreLinkDefaultPipeline(level);
                }
                else
                {
                    mpm = passBuilder.buildPerModuleDefaultPipeline(level);
                }
            }

            if (args.hasArg(OPT_emit_llvm) || lto)
            {
                // See
                // https://github.com/llvm/llvm-project/blob/ea22fdd120aeb1bbb9ea96670d70193dc02b2c5f/clang/lib/CodeGen/BackendUtil.cpp#L1467
                // Doing full LTO for now
                bool emitLTOSummary = lto && m_targetMachine->getTargetTriple().getVendor() != llvm::Triple::Apple;
                if (emitLTOSummary)
                {
                    if (!llvmModule->getModuleFlag("ThinLTO"))
                    {
                        llvmModule->addModuleFlag(llvm::Module::Error, "ThinLTO", std::uint32_t(0));
                    }
                    if (!llvmModule->getModuleFlag("EnableSplitLTOUnit"))
                    {
                        llvmModule->addModuleFlag(llvm::Module::Error, "EnableSplitLTOUnit", std::uint32_t(1));
                    }
                }
                if (mlir::failed(ensureOutputStream(args, action)))
                {
                    return mlir::failure();
                }
                if (action == pylir::CompilerInvocation::Assembly)
                {
                    mpm.addPass(llvm::PrintModulePass(*m_output));
                }
                else
                {
                    mpm.addPass(llvm::BitcodeWriterPass(*m_output, false, emitLTOSummary));
                }
            }

            mpm.run(*llvmModule, mam);

            if (args.hasArg(OPT_emit_llvm))
            {
                return finalizeOutputStream(mlir::success());
            }
            if (lto)
            {
                break;
            }
            if (mlir::failed(ensureOutputStream(args, action)))
            {
                return mlir::failure();
            }

            llvm::legacy::PassManager codeGenPasses;
            codeGenPasses.add(llvm::createTargetTransformInfoWrapperPass(m_targetMachine->getTargetIRAnalysis()));
            if (m_targetMachine->addPassesToEmitFile(
                    codeGenPasses, *m_output, nullptr,
                    action == pylir::CompilerInvocation::Assembly ? llvm::CGFT_AssemblyFile : llvm::CGFT_ObjectFile))
            {
                std::string_view format = action == pylir::CompilerInvocation::Assembly ? "Assembly" : "Object file";
                auto* arg = args.getLastArg(OPT_target_EQ);
                if (!arg)
                {
                    arg = args.getLastArg(OPT_c, OPT_S);
                }
                if (arg)
                {
                    llvm::errs() << commandLine
                                        .createDiagnosticsBuilder(arg,
                                                                  pylir::Diag::TARGET_N_DOES_NOT_SUPPORT_COMPILING_TO_N,
                                                                  m_targetMachine->getTargetTriple().str(), format)
                                        .addLabel(arg, std::nullopt, pylir::Diag::ERROR_COLOUR)
                                        .emitError();
                    return finalizeOutputStream(mlir::failure());
                }
                llvm::errs() << pylir::Diag::formatLine(
                    pylir::Diag::Error, fmt::format(pylir::Diag::TARGET_N_DOES_NOT_SUPPORT_COMPILING_TO_N,
                                                    m_targetMachine->getTargetTriple().str(), format));
                return finalizeOutputStream(mlir::failure());
            }

            codeGenPasses.run(*llvmModule);
            break;
        }
    }
    if (action != Link)
    {
        return finalizeOutputStream(mlir::success());
    }
    auto fileName = m_outputFile->TmpName;
    m_outFileStream.reset();
    if (auto error = m_outputFile->keep())
    {
        llvm::consumeError(std::move(error));
        llvm::errs() << pylir::Diag::formatLine(
            pylir::Diag::Error, fmt::format(pylir::Diag::FAILED_TO_KEEP_TEMPORARY_FILE_N, m_outputFile->TmpName));
        return mlir::failure();
    }
    bool success = toolchain.link(commandLine, fileName);
    llvm::sys::fs::remove(fileName);
    return mlir::success(success);
}

void pylir::CompilerInvocation::ensureMLIRContext(const llvm::opt::InputArgList& args)
{
    if (m_mlirContext)
    {
        return;
    }
    mlir::DialectRegistry registry;
    registry.insert<pylir::Py::PylirPyDialect>();
    registry.insert<pylir::Mem::PylirMemDialect>();
    registry.insert<mlir::StandardOpsDialect>();
    registry.insert<mlir::arith::ArithmeticDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
    registry.insert<mlir::cf::ControlFlowDialect>();
    m_mlirContext.emplace(registry);
    m_mlirContext->enableMultithreading(args.hasArg(OPT_Xmulti_threaded, OPT_Xsingle_threaded, true));
    m_mlirContext->getDiagEngine().registerHandler(
        [](mlir::Diagnostic& diagnostic)
        {
            diagnostic.print(llvm::errs());
            llvm::errs() << '\n';
        });
}

mlir::LogicalResult pylir::CompilerInvocation::ensureOutputStream(const llvm::opt::InputArgList& args, Action action)
{
    if (m_output)
    {
        return mlir::success();
    }
    auto filename = llvm::sys::path::filename(args.getLastArgValue(OPT_INPUT)).str();
    llvm::SmallString<20> realOutputFilename;
    if (action == pylir::CompilerInvocation::Link)
    {
        llvm::sys::path::system_temp_directory(true, realOutputFilename);
        llvm::sys::path::append(realOutputFilename, "tmp.o");
    }
    else
    {
        std::string defaultName;
        if (args.hasArg(OPT_emit_mlir))
        {
            defaultName = filename + ".mlir";
        }
        else if (args.hasArg(OPT_emit_llvm))
        {
            if (action == pylir::CompilerInvocation::ObjectFile)
            {
                defaultName = filename + ".bc";
            }
            else
            {
                defaultName = filename + ".ll";
            }
        }
        else if (action == pylir::CompilerInvocation::Assembly)
        {
            defaultName = filename + ".s";
        }
        else if (action == pylir::CompilerInvocation::ObjectFile)
        {
            defaultName = filename + ".o";
        }
        realOutputFilename = args.getLastArgValue(OPT_o, defaultName);
    }

    llvm::SmallString<20> tempFileName = realOutputFilename;
    if (realOutputFilename == "-")
    {
        m_output = &llvm::outs();
        return mlir::success();
    }
    auto extension = llvm::sys::path::extension(realOutputFilename);
    llvm::sys::path::remove_filename(tempFileName);
    llvm::sys::path::append(tempFileName, llvm::sys::path::stem(realOutputFilename) + "-%%%%" + extension);
    auto tempFile = llvm::sys::fs::TempFile::create(tempFileName);
    if (!tempFile)
    {
        llvm::consumeError(tempFile.takeError());
        llvm::errs() << pylir::Diag::formatLine(
            pylir::Diag::Error, fmt::format(pylir::Diag::FAILED_TO_CREATE_TEMPORARY_FILE_N, tempFileName.str()));
        return mlir::failure();
    }
    m_outputFile = std::move(*tempFile);
    m_outFileStream.emplace(m_outputFile->FD, false);
    m_output = &*m_outFileStream;
    m_realOutputFilename = realOutputFilename.str();
    return mlir::success();
}

mlir::LogicalResult pylir::CompilerInvocation::finalizeOutputStream(mlir::LogicalResult result)
{
    if (!m_outputFile)
    {
        return result;
    }
    m_outFileStream.reset();
    m_output = nullptr;
    auto exit = llvm::make_scope_exit([&] { m_outputFile.reset(); });
    if (mlir::failed(result))
    {
        if (auto error = m_outputFile->discard())
        {
            llvm::consumeError(std::move(error));
            llvm::errs() << pylir::Diag::formatLine(
                pylir::Diag::Error,
                fmt::format(pylir::Diag::FAILED_TO_DISCARD_TEMPORARY_FILE_N, m_outputFile->TmpName));
        }
        return mlir::failure();
    }
    if (auto error = m_outputFile->keep(m_realOutputFilename))
    {
        llvm::consumeError(std::move(error));
        llvm::errs() << pylir::Diag::formatLine(pylir::Diag::Error,
                                                fmt::format(pylir::Diag::FAILED_TO_RENAME_TEMPORARY_FILE_N_TO_N,
                                                            m_outputFile->TmpName, m_realOutputFilename));
        return mlir::failure();
    }
    return mlir::success();
}

void pylir::CompilerInvocation::addOptimizationPasses(llvm::StringRef level, mlir::OpPassManager& manager)
{
    manager.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
    if (level != "0")
    {
        manager.addNestedPass<mlir::FuncOp>(pylir::Py::createHandleLoadStoreEliminationPass());
        manager.addPass(pylir::Py::createFoldHandlesPass());
        manager.addNestedPass<mlir::FuncOp>(mlir::createCSEPass());
        manager.addNestedPass<mlir::FuncOp>(mlir::createSCCPPass());
        manager.addNestedPass<mlir::FuncOp>(pylir::createLoadForwardingPass());
    }
    manager.addPass(pylir::Py::createExpandPyDialectPass());
    if (level != "0")
    {
        manager.addPass(mlir::createCanonicalizerPass());
        manager.addNestedPass<mlir::FuncOp>(mlir::createCSEPass());
        manager.addNestedPass<mlir::FuncOp>(mlir::createSCCPPass());
        manager.addNestedPass<mlir::FuncOp>(pylir::createLoadForwardingPass());
    }
    manager.addPass(pylir::createConvertPylirPyToPylirMemPass());
}

mlir::LogicalResult pylir::CompilerInvocation::ensureTargetMachine(const llvm::opt::InputArgList& args,
                                                                   const cli::CommandLine& commandLine,
                                                                   const pylir::Toolchain& toolchain,
                                                                   llvm::Optional<llvm::Triple> triple)
{
    if (mlir::failed(ensureLLVMInit(args)))
    {
        return mlir::failure();
    }
    if (m_targetMachine)
    {
        return mlir::success();
    }
    if (!triple)
    {
        triple = llvm::Triple(args.getLastArgValue(OPT_target_EQ, LLVM_DEFAULT_TARGET_TRIPLE));
    }
    std::string error;
    const auto* targetM = llvm::TargetRegistry::lookupTarget(triple->str(), error);
    if (!targetM)
    {
        auto* outputArg = args.getLastArg(OPT_target_EQ);
        if (!outputArg)
        {
            llvm::errs() << pylir::Diag::formatLine(pylir::Diag::Error,
                                                    fmt::format(pylir::Diag::COULD_NOT_FIND_TARGET_N, triple->str()));
            return mlir::failure();
        }
        llvm::errs() << commandLine
                            .createDiagnosticsBuilder(outputArg, pylir::Diag::COULD_NOT_FIND_TARGET_N, triple->str())
                            .addLabel(outputArg, std::nullopt, pylir::Diag::ERROR_COLOUR)
                            .emitError();
        return mlir::failure();
    }

    auto optLevel = llvm::StringSwitch<std::optional<llvm::CodeGenOpt::Level>>(args.getLastArgValue(OPT_O, "0"))
                        .Case("0", llvm::CodeGenOpt::None)
                        .Case("1", llvm::CodeGenOpt::Less)
                        .Case("2", llvm::CodeGenOpt::Default)
                        .Case("3", llvm::CodeGenOpt::Aggressive)
                        .Case("4", llvm::CodeGenOpt::Aggressive)
                        .Case("s", llvm::CodeGenOpt::Default)
                        .Case("z", llvm::CodeGenOpt::Default)
                        .Default(std::nullopt);
    if (!optLevel)
    {
        auto* optArg = args.getLastArg(OPT_O);
        llvm::errs() << commandLine
                            .createDiagnosticsBuilder(optArg, pylir::Diag::INVALID_OPTIMIZATION_LEVEL_N,
                                                      optArg->getAsString(args))
                            .addLabel(optArg, std::nullopt, pylir::Diag::ERROR_COLOUR)
                            .emitError();
        return mlir::failure();
    }

    llvm::Reloc::Model relocation = llvm::Reloc::Static;
    if (toolchain.defaultsToPIC() || toolchain.isPIE(commandLine))
    {
        relocation = llvm::Reloc::PIC_;
    }
    llvm::TargetOptions targetOptions;
    if (triple->isOSWindows())
    {
        targetOptions.ExceptionModel = llvm::ExceptionHandling::WinEH;
    }
    else
    {
        targetOptions.ExceptionModel = llvm::ExceptionHandling::DwarfCFI;
    }
    targetOptions.UseInitArray = true;
    m_targetMachine = std::unique_ptr<llvm::TargetMachine>(
        targetM->createTargetMachine(triple->str(), "generic", "", targetOptions, relocation, {}, *optLevel));
    return mlir::success();
}

mlir::LogicalResult pylir::CompilerInvocation::ensureLLVMInit(const llvm::opt::InputArgList& args)
{
    if (m_llvmContext)
    {
        return mlir::success();
    }
    m_llvmContext = std::make_unique<llvm::LLVMContext>();
    std::vector<const char*> refs;
    refs.push_back("pylir (LLVM option parsing)");

    // Allow callee saved registers for live-through and GC ptr values
    refs.push_back("-fixup-allow-gcptr-in-csr");
    if (args.getLastArgValue(OPT_O, "0") != "0")
    {
        // TODO: I am intending this as a suggestion to the register allocator, not a requirement. In O0 this will lead
        //       to "out of registers" errors however. This might also just be a pass ordering issue or something else
        //       however as the FixupStatepointCallerSaved is not ran
        // No restrictions on how many registers its allowed to use
        refs.push_back("-max-registers-for-gc-values=1000");
    }

    auto options = args.getAllArgValues(OPT_mllvm);
    std::transform(options.begin(), options.end(), std::back_inserter(refs),
                   [](const std::string& str) { return str.c_str(); });
    refs.push_back(nullptr);
    return mlir::success(llvm::cl::ParseCommandLineOptions(refs.size() - 1, refs.data(), "", &llvm::errs()));
}
