// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "CompilerInvocation.hpp"

#include <mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Arithmetic/Transforms/Passes.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/Transforms/LegalizeForExport.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
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
#include <llvm/Support/Program.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ThreadPool.h>

#include <pylir/CodeGen/CodeGen.hpp>
#include <pylir/Diagnostics/DiagnosticMessages.hpp>
#include <pylir/LLVM/PlaceStatepoints.hpp>
#include <pylir/LLVM/PylirGC.hpp>
#include <pylir/Optimizer/Conversion/Passes.hpp>
#include <pylir/Optimizer/Linker/Linker.hpp>
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
    // --lld-path overrides -f[no-]integrated-lld unconditionally. If the embedded lld
    // is used and -O4 enable LTO
    bool enable = !args.hasArg(OPT_lld_path_EQ) && args.getLastArgValue(OPT_O, "0") == "4"
                  && args.hasFlag(OPT_fintegrated_lld, OPT_fno_integrated_lld, true);
    if (commandLine.verbose())
    {
        if (enable)
        {
            llvm::errs() << "Enabling LTO as integrated LLD and -O4 was enabled\n";
        }
        else if (args.getLastArgValue(OPT_O, "0") == "4"
                 && (args.hasArg(OPT_lld_path_EQ) || !args.hasFlag(OPT_fintegrated_lld, OPT_fno_integrated_lld, true)))
        {
            llvm::errs() << "LTO not enabled as integrated LLD is not used. Add '-flto' if your linker supports LTO\n";
        }
    }
    return enable;
#else
    return false;
#endif
}

// https://cmake.org/cmake/help/latest/command/add_custom_command.html for format
llvm::SmallString<100> escapeForMakefile(llvm::StringRef filename)
{
    llvm::SmallString<100> result;
    for (char c : filename)
    {
        switch (c)
        {
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

void writeDependencyOutput(llvm::raw_ostream& ostream, llvm::StringRef outputFilename,
                           llvm::ArrayRef<std::string> additionalInputFiles)
{
    ostream << escapeForMakefile(outputFilename) << ": ";
    llvm::interleave(llvm::map_range(additionalInputFiles, escapeForMakefile), ostream, " ");
    ostream << '\n';
}
} // namespace

mlir::LogicalResult pylir::CompilerInvocation::executeAction(llvm::opt::Arg* inputFile,
                                                             const cli::CommandLine& commandLine,
                                                             const pylir::Toolchain& toolchain,
                                                             CompilerInvocation::Action action)
{
    const auto& args = commandLine.getArgs();
    std::optional<llvm::ToolOutputFile> outputFile;
    if (!commandLine.onlyPrint())
    {
        if (auto* arg = args.getLastArg(OPT_M))
        {
            std::error_code ec;
            outputFile.emplace(arg->getValue(), ec, llvm::sys::fs::OF_Text);
            if (ec)
            {
                llvm::errs() << commandLine.createDiagnosticsBuilder(arg, Diag::FAILED_TO_OPEN_FILE_N, arg->getValue())
                                    .addLabel(arg, std::nullopt, Diag::ERROR_COLOUR)
                                    .emitError();
                return mlir::failure();
            }
            if (arg = args.getLastArg(OPT_o); arg && arg->getValue() == llvm::StringRef{"-"})
            {
                llvm::errs() << commandLine
                                    .createDiagnosticsBuilder(
                                        arg, Diag::OUTPUT_CANNOT_BE_STDOUT_WHEN_WRITING_DEPENDENCY_FILE)
                                    .addLabel(arg, std::nullopt, Diag::ERROR_COLOUR)
                                    .emitError();
                return mlir::failure();
            }
        }
        if (mlir::failed(compilation(inputFile, commandLine, toolchain, action)))
        {
            return mlir::failure();
        }
    }
    if (outputFile)
    {
        // The very first document is the main input file which build systems already depend on. Hence, there is no
        // need to add it to the output.
        std::vector<std::string> additionalInputFiles(m_documents.size() - 1);
        llvm::transform(llvm::drop_begin(m_documents), additionalInputFiles.begin(),
                        [](const Diag::Document& document) { return document.getFilename(); });

        // The document order is non-deterministic when multi threading is enabled. Sort them to make it deterministic
        // again.
        std::sort(additionalInputFiles.begin(), additionalInputFiles.end());
        writeDependencyOutput(outputFile->os(), m_actionOutputFilename, additionalInputFiles);
        outputFile->keep();
    }
    if (action != Link)
    {
        return finalizeOutputStream(mlir::success());
    }
    if (commandLine.onlyPrint())
    {
        if (mlir::failed(ensureOutputStream(args, action)))
        {
            return mlir::failure();
        }
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

mlir::LogicalResult pylir::CompilerInvocation::compilation(llvm::opt::Arg* inputFile,
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

    auto shouldOutput = [&args](pylir::cli::ID id)
    {
        auto* outputFormatArg = args.getLastArg(OPT_emit_llvm, OPT_emit_mlir, OPT_emit_pylir);
        return outputFormatArg && outputFormatArg->getOption().getID() == id;
    };

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
            auto& document = m_documents.emplace_back(std::move(content), inputFile->getValue());
            Syntax::FileInput* fileInput;
            {
                pylir::Parser parser(document);
                auto tree = parser.parseFileInput();
                if (!tree)
                {
                    llvm::errs() << tree.error();
                    return mlir::failure();
                }
                fileInput = &m_fileInputs.emplace_back(std::move(*tree));
            }
            if (args.hasArg(OPT_dump_ast))
            {
                pylir::Dumper dumper;
                llvm::outs() << dumper.dump(*fileInput);
            }
            if (action == SyntaxOnly)
            {
                return mlir::success();
            }
            ensureMLIRContext(args);

            auto module = codegenPythonToMLIR(args, commandLine);
            if (mlir::failed(module))
            {
                return mlir::failure();
            }
            mlirModule = std::move(*module);
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
            if (mlir::failed(mlir::verify(*mlirModule)))
            {
                return mlir::failure();
            }
    #if !defined(__MINGW32_MAJOR_VERSION) || !defined(__clang__)
            manager.enableCrashReproducerGeneration("failure.mlir");
    #endif
#else
            manager.enableVerifier(false);
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
                                         args.hasArg(OPT_Xprint_module_scope), false, false, llvm::errs(),
                                         mlir::OpPrintingFlags().enableDebugInfo().assumeVerified());
            }
            if (args.hasArg(OPT_Xtiming))
            {
                manager.enableTiming();
            }
            if (shouldOutput(OPT_emit_pylir))
            {
                if (mlir::failed(manager.run(*mlirModule)))
                {
                    return mlir::failure();
                }
                if (mlir::failed(ensureOutputStream(args, action)))
                {
                    return mlir::failure();
                }
                mlirModule->print(*m_output, mlir::OpPrintingFlags{}.assumeVerified().enableDebugInfo());
                return finalizeOutputStream(mlir::success());
            }
            addOptimizationPasses(args.getLastArgValue(OPT_O, "0"), manager);
            if (shouldOutput(OPT_emit_mlir))
            {
                if (mlir::failed(manager.run(*mlirModule)))
                {
                    return mlir::failure();
                }
                if (mlir::failed(ensureOutputStream(args, action)))
                {
                    return mlir::failure();
                }
                mlirModule->print(*m_output, mlir::OpPrintingFlags{}.assumeVerified().enableDebugInfo());
                return finalizeOutputStream(mlir::success());
            }
            if (mlir::failed(ensureTargetMachine(args, commandLine, toolchain)))
            {
                return mlir::failure();
            }
            auto* nested = &manager.nestAny();
            nested->addPass(mlir::arith::createArithmeticExpandOpsPass());
            nested->addPass(mlir::arith::createConvertArithmeticToLLVMPass());
            manager.addPass(pylir::createConvertPylirToLLVMPass(m_targetMachine->getTargetTriple(),
                                                                m_targetMachine->createDataLayout()));
            nested = &manager.nestAny();
            nested->addPass(mlir::createReconcileUnrealizedCastsPass());
            nested->addPass(mlir::LLVM::createLegalizeForExportPass());
            if (args.hasArg(OPT_Xprint_pipeline))
            {
                manager.printAsTextualPipeline(llvm::errs());
                llvm::errs() << '\n';
            }
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
                if (mlir::failed(ensureLLVMInit(args, toolchain)))
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

            passBuilder.registerOptimizerLastEPCallback([](llvm::ModulePassManager& mpm, llvm::OptimizationLevel)
                                                        { mpm.addPass(pylir::PlaceStatepointsPass{}); });

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

            if (shouldOutput(OPT_emit_llvm) || lto)
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

            if (shouldOutput(OPT_emit_llvm))
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
    return mlir::success();
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
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithmeticDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
    registry.insert<mlir::cf::ControlFlowDialect>();
    m_mlirContext.emplace(registry);
    m_mlirContext->enableMultithreading(args.hasFlag(OPT_Xmulti_threaded, OPT_Xsingle_threaded, true));
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
    m_compileStepOutputFilename = realOutputFilename.str();
    m_actionOutputFilename = action == Link ? args.getLastArgValue(OPT_o) : m_compileStepOutputFilename;
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
    if (auto error = m_outputFile->keep(m_compileStepOutputFilename))
    {
        llvm::consumeError(std::move(error));
        llvm::errs() << pylir::Diag::formatLine(pylir::Diag::Error,
                                                fmt::format(pylir::Diag::FAILED_TO_RENAME_TEMPORARY_FILE_N_TO_N,
                                                            m_outputFile->TmpName, m_compileStepOutputFilename));
        return mlir::failure();
    }
    return mlir::success();
}

void pylir::CompilerInvocation::addOptimizationPasses(llvm::StringRef level, mlir::OpPassManager& manager)
{
    mlir::OpPassManager* nested;
    manager.addPass(mlir::createCanonicalizerPass());
    if (level != "0")
    {
        manager.nestAny().addPass(pylir::Py::createHandleLoadStoreEliminationPass());
        manager.addPass(pylir::Py::createFoldHandlesPass());
        manager.nestAny().addPass(mlir::createCSEPass());
        manager.addPass(pylir::Py::createTrialInlinerPass());
        manager.addPass(mlir::createSymbolDCEPass());
        nested = &manager.nestAny();
        nested->addPass(pylir::createLoadForwardingPass());
        nested->addPass(mlir::createSCCPPass());
        manager.addPass(pylir::Py::createMonomorphPass());
        manager.addPass(pylir::Py::createTrialInlinerPass());
        manager.addPass(mlir::createSymbolDCEPass());
    }
    nested = &manager.nestAny();
    nested->addPass(pylir::Py::createExpandPyDialectPass());
    if (level != "0")
    {
        nested->addPass(mlir::createCanonicalizerPass());
        nested->addPass(mlir::createCSEPass());
        nested->addPass(pylir::createLoadForwardingPass());
        nested->addPass(mlir::createSCCPPass());
    }
    manager.addPass(pylir::createConvertPylirPyToPylirMemPass());
}

mlir::LogicalResult pylir::CompilerInvocation::ensureTargetMachine(const llvm::opt::InputArgList& args,
                                                                   const cli::CommandLine& commandLine,
                                                                   const pylir::Toolchain& toolchain,
                                                                   llvm::Optional<llvm::Triple> triple)
{
    if (mlir::failed(ensureLLVMInit(args, toolchain)))
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

mlir::LogicalResult pylir::CompilerInvocation::ensureLLVMInit(const llvm::opt::InputArgList& args,
                                                              const pylir::Toolchain& toolchain)
{
    if (m_llvmContext)
    {
        return mlir::success();
    }
    m_llvmContext = std::make_unique<llvm::LLVMContext>();
    m_llvmContext->setOpaquePointers(true);
    std::vector<const char*> refs;
    refs.push_back("pylir (LLVM option parsing)");

    auto options = toolchain.getLLVMOptions(args);
    std::transform(options.begin(), options.end(), std::back_inserter(refs),
                   [](const std::string& str) { return str.c_str(); });
    refs.push_back(nullptr);
    llvm::cl::ResetAllOptionOccurrences();
    return mlir::success(llvm::cl::ParseCommandLineOptions(refs.size() - 1, refs.data(), "", &llvm::errs()));
}

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
    pylir::CompilerInvocation::codegenPythonToMLIR(const llvm::opt::InputArgList& args,
                                                   const cli::CommandLine& commandLine)
{
    pylir::CodeGenOptions options{};
    std::vector<std::string> importPaths;
    {
        llvm::SmallString<100> docPath{m_documents.front().getFilename()};
        llvm::sys::fs::make_absolute(docPath);
        llvm::sys::path::remove_filename(docPath);
        importPaths.emplace_back(docPath);
    }
    for (auto& iter : args.getAllArgValues(OPT_I))
    {
        llvm::SmallString<100> path{iter};
        llvm::sys::fs::make_absolute(path);
        importPaths.emplace_back(path);
    }

    auto appendEnvVar = [&](llvm::StringRef value)
    {
        llvm::SmallVector<llvm::StringRef> output;
        value.split(output, llvm::sys::EnvPathSeparator);
        llvm::transform(output, std::back_inserter(importPaths),
                        [](llvm::SmallString<100> path)
                        {
                            llvm::sys::fs::make_absolute(path);
                            return std::string{path};
                        });
    };

    if (auto* pythonPath = std::getenv("PYTHONPATH"))
    {
        appendEnvVar(pythonPath);
    }
    if (auto* pythonHome = std::getenv("PYTHONHOME"))
    {
        appendEnvVar(pythonHome);
    }
    else
    {
        llvm::SmallString<100> libDirPath = commandLine.getExecutablePath();
        llvm::sys::path::remove_filename(libDirPath);
        llvm::sys::path::append(libDirPath, "..", "lib");
        importPaths.emplace_back(libDirPath);
    }
    options.importPaths = std::move(importPaths);

    // Protects 'futures' and 'loaded'.
    std::mutex dataStructureMutex;
    // std::list is used deliberately here as we need iteration stability on push_back. This is important for the
    // case where multi threading is disabled: We are used deferred launches in futures to have as compatible of
    // interfaces as possible and these are only launched when the item to be computed is retrieved. That retrieval
    // may however import more item and append to 'futures'. Since we are already mid-iteration, this has to be
    // possible.
    std::list<std::pair<std::shared_future<mlir::ModuleOp>, std::string>> futures;
    llvm::StringSet<> loaded;

    // Protects output to 'llvm::errs()'.
    std::mutex outputMutex;
    // Protects 'm_fileInputs' and 'm_documents'.
    std::mutex sourceDSMutex;

    options.warningCallback = [&](Diag::DiagnosticsBuilder&& builder)
    {
        std::unique_lock lock{outputMutex};
        llvm::errs() << builder.emitWarning();
    };

    options.moduleLoadCallback = [&](CodeGenOptions::LoadRequest&& request)
    {
        std::unique_lock lock{dataStructureMutex};
        auto qualifier = request.qualifier;
        if (!loaded.insert(qualifier).second)
        {
            return;
        }

        auto action = [&, request = std::move(request)]() mutable -> mlir::ModuleOp
        {
            std::optional exit = llvm::make_scope_exit([&] { llvm::sys::fs::closeFile(request.handle); });
            llvm::sys::fs::file_status status;
            {
                auto error = llvm::sys::fs::status(request.handle, status);
                if (error)
                {
                    std::unique_lock lock{outputMutex};
                    llvm::errs() << Diag::DiagnosticsBuilder(*request.document, request.location,
                                                             pylir::Diag::FAILED_TO_ACCESS_FILE_N, request.filePath)
                                        .addLabel(request.location, std::nullopt, pylir::Diag::ERROR_COLOUR)
                                        .emitError();
                    return nullptr;
                }
            }
            std::string content(status.getSize(), '\0');
            auto read = llvm::sys::fs::readNativeFile(request.handle, {content.data(), content.size()});
            if (!read)
            {
                std::unique_lock lock{outputMutex};
                llvm::errs() << Diag::DiagnosticsBuilder(*request.document, request.location,
                                                         pylir::Diag::FAILED_TO_READ_FILE_N, request.filePath)
                                    .addLabel(request.location, std::nullopt, pylir::Diag::ERROR_COLOUR)
                                    .emitError();
                return nullptr;
            }
            exit.reset();

            std::unique_lock sourceLock{sourceDSMutex};
            auto& document = m_documents.emplace_back(std::move(content), request.filePath);
            sourceLock.unlock();

            pylir::Parser parser(document);
            auto tree = parser.parseFileInput();
            if (!tree)
            {
                std::unique_lock lock{outputMutex};
                llvm::errs() << tree.error();
                return nullptr;
            }

            sourceLock.lock();
            auto& fileInput = m_fileInputs.emplace_back(std::move(*tree));
            sourceLock.unlock();

            auto copyOption = options;
            copyOption.qualifier = std::move(request.qualifier);
            return pylir::codegen(&*m_mlirContext, fileInput, document, copyOption).release();
        };

        if (m_mlirContext->isMultithreadingEnabled())
        {
            futures.emplace_back(m_mlirContext->getThreadPool().async(std::move(action)), std::move(qualifier));
        }
        else
        {
            futures.emplace_back(std::async(std::launch::deferred, std::move(action)).share(), std::move(qualifier));
        }
    };

    auto mainModule = pylir::codegen(&*m_mlirContext, m_fileInputs.front(), m_documents.front(), options);
    std::vector<mlir::OwningOpRef<mlir::ModuleOp>> importedModules;
    importedModules.push_back(std::move(mainModule));
    // The real size of `futures` is unknown as it grows while we are iterating through here. Hence, we NEED to use
    // a back inserter.
    std::vector<std::pair<mlir::OwningOpRef<mlir::ModuleOp>, std::string>> calculatedImports;
    std::transform(std::move_iterator(futures.begin()), std::move_iterator(futures.end()),
                   std::back_inserter(calculatedImports),
                   [](auto&& pair) {
                       return std::pair{pair.first.get(), std::move(pair.second)};
                   });
    if (!llvm::all_of(llvm::make_first_range(calculatedImports), llvm::identity<mlir::OwningOpRef<mlir::ModuleOp>>{}))
    {
        return mlir::failure();
    }
    std::sort(calculatedImports.begin(), calculatedImports.end(), llvm::less_second{});
    std::transform(std::move_iterator(calculatedImports.begin()), std::move_iterator(calculatedImports.end()),
                   std::back_inserter(importedModules), [](auto&& pair) { return std::move(pair.first); });

    return linkModules(importedModules);
}
