
#include "PylirMain.hpp"

#include <mlir/Dialect/LLVMIR/Transforms/LegalizeForExport.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/Triple.h>
#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Bitcode/BitcodeWriterPass.h>
#include <llvm/IR/IRPrintingPasses.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Option/Arg.h>
#include <llvm/Option/ArgList.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>

#include <pylir/CodeGen/CodeGen.hpp>
#include <pylir/Diagnostics/DiagnosticMessages.hpp>
#include <pylir/Main/Opts.inc>
#include <pylir/Optimizer/Conversion/Passes.hpp>
#include <pylir/Optimizer/PylirPy/Transform/Passes.hpp>
#include <pylir/Parser/Dumper.hpp>
#include <pylir/Parser/Parser.hpp>

#include "CommandLine.hpp"
#include "LinuxToolchain.hpp"
#include "MSVCToolchain.hpp"
#include "MinGWToolchain.hpp"
#include "Toolchain.hpp"

using namespace pylir::cli;

namespace
{

enum class Action
{
    SyntaxOnly,
    ObjectFile,
    Assembly,
    Link
};

} // namespace

namespace
{

bool enableLTO(const pylir::cli::CommandLine& commandLine)
{
    auto& args = commandLine.getArgs();
    if (args.hasFlag(OPT_flto, OPT_fno_lto, false))
    {
        return true;
    }
#ifdef PYLIR_EMBEDDED_LLD
    // --ld-path overrides -f[no-]integrated-ld unconditionally. If the embedded ld
    // is used and -O3 enable LTO
    return !args.hasArg(OPT_ld_path_EQ) && args.getLastArgValue(OPT_O, "0") == "3"
           && args.hasFlag(OPT_fintegrated_ld, OPT_fno_integrated_ld, true);
#else
    return false;
#endif
}

bool executeCompilation(Action action, const pylir::Syntax::FileInput& tree, pylir::Diag::Document& file,
                        const pylir::cli::CommandLine& commandLine, llvm::raw_pwrite_stream& output)
{
    auto& options = commandLine.getArgs();
    mlir::MLIRContext context;
    context.getDiagEngine().registerHandler([](mlir::Diagnostic& diagnostic) { diagnostic.print(llvm::errs()); });
    auto module = pylir::codegen(&context, tree, file);

    mlir::PassManager manager(&context);
#ifndef NDEBUG
    manager.enableVerifier();
    #if !defined(__MINGW32_MAJOR_VERSION) || !defined(__clang__)
    manager.enableCrashReproducerGeneration("failure.mlir");
    #endif
    manager.enableIRPrinting(std::make_unique<mlir::PassManager::IRPrinterConfig>(false, false, true));
#endif
    if (options.hasArg(OPT_emit_pylir))
    {
        if (mlir::failed(manager.run(*module)))
        {
            return false;
        }
        module->print(output, mlir::OpPrintingFlags{}.enableDebugInfo());
        return true;
    }
    if (options.getLastArgValue(OPT_O, "0") != "0")
    {
        manager.addPass(mlir::createCanonicalizerPass());
        manager.addNestedPass<mlir::FuncOp>(mlir::createCSEPass());
        manager.addNestedPass<mlir::FuncOp>(mlir::createSCCPPass());
    }
    manager.addNestedPass<mlir::FuncOp>(pylir::Py::createExpandPyDialectPass());
    if (options.getLastArgValue(OPT_O, "0") != "0")
    {
        manager.addPass(mlir::createCanonicalizerPass());
        manager.addNestedPass<mlir::FuncOp>(mlir::createCSEPass());
        manager.addNestedPass<mlir::FuncOp>(mlir::createSCCPPass());
    }
    manager.addPass(pylir::createConvertPylirPyToPylirMemPass());
    if (options.hasArg(OPT_emit_mlir))
    {
        if (mlir::failed(manager.run(*module)))
        {
            return false;
        }
        module->print(output, mlir::OpPrintingFlags{}.enableDebugInfo());
        return true;
    }

    auto triple = llvm::Triple(options.getLastArgValue(OPT_target_EQ, LLVM_DEFAULT_TARGET_TRIPLE));
    std::string error;
    auto* targetM = llvm::TargetRegistry::lookupTarget(triple.str(), error);
    if (!targetM)
    {
        auto outputArg = options.getLastArg(OPT_target_EQ);
        if (!outputArg)
        {
            llvm::errs() << pylir::Diag::formatLine(pylir::Diag::Error,
                                                    fmt::format(pylir::Diag::COULD_NOT_FIND_TARGET_N, triple.str()));
            return false;
        }
        llvm::errs() << commandLine
                            .createDiagnosticsBuilder(outputArg, pylir::Diag::COULD_NOT_FIND_TARGET_N, triple.str())
                            .addLabel(outputArg, std::nullopt, pylir::Diag::ERROR_COLOUR)
                            .emitError();
        return false;
    }

    auto optLevel = llvm::StringSwitch<std::optional<llvm::CodeGenOpt::Level>>(options.getLastArgValue(OPT_O, "0"))
                        .Case("0", llvm::CodeGenOpt::None)
                        .Case("1", llvm::CodeGenOpt::Less)
                        .Case("2", llvm::CodeGenOpt::Default)
                        .Case("3", llvm::CodeGenOpt::Aggressive)
                        .Case("s", llvm::CodeGenOpt::Default)
                        .Case("z", llvm::CodeGenOpt::Default)
                        .Default(std::nullopt);
    if (!optLevel)
    {
        auto optArg = options.getLastArg(OPT_O);
        llvm::errs() << commandLine
                            .createDiagnosticsBuilder(optArg, pylir::Diag::INVALID_OPTIMIZATION_LEVEL_N,
                                                      optArg->getAsString(options))
                            .addLabel(optArg, std::nullopt, pylir::Diag::ERROR_COLOUR)
                            .emitError();
        return false;
    }

    llvm::Reloc::Model relocation = llvm::Reloc::Static;
    if (triple.isOSWindows() && triple.getArch() == llvm::Triple::x86_64)
    {
        relocation = llvm::Reloc::PIC_;
    }
    auto machine = std::unique_ptr<llvm::TargetMachine>(
        targetM->createTargetMachine(triple.str(), "generic", "", {}, relocation, {}, *optLevel));

    std::string passOptions =
        "target-triple=" + triple.str() + " data-layout=" + machine->createDataLayout().getStringRepresentation();

    auto pass = pylir::Mem::createConvertPylirToLLVMPass();
    if (mlir::failed(pass->initializeOptions(passOptions)))
    {
        return false;
    }

    manager.addPass(std::move(pass));
    manager.addPass(mlir::LLVM::createLegalizeForExportPass());
    if (mlir::failed(manager.run(*module)))
    {
        return false;
    }

    llvm::LLVMContext llvmContext;
    mlir::registerLLVMDialectTranslation(context);
    auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);

    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;
    llvm::PassBuilder passBuilder(machine.get());

    fam.registerPass([&] { return passBuilder.buildDefaultAAPipeline(); });
    passBuilder.registerModuleAnalyses(mam);
    passBuilder.registerCGSCCAnalyses(cgam);
    passBuilder.registerFunctionAnalyses(fam);
    passBuilder.registerLoopAnalyses(lam);
    passBuilder.crossRegisterProxies(lam, fam, cgam, mam);

    bool lto = enableLTO(commandLine);
    llvm::ModulePassManager mpm;
    if (options.getLastArgValue(OPT_O, "0") == "0")
    {
        mpm = passBuilder.buildO0DefaultPipeline(llvm::OptimizationLevel::O0);
    }
    else
    {
        llvm::OptimizationLevel level = llvm::StringSwitch<llvm::OptimizationLevel>(options.getLastArgValue(OPT_O))
                                            .Case("1", llvm::OptimizationLevel::O1)
                                            .Case("2", llvm::OptimizationLevel::O2)
                                            .Case("3", llvm::OptimizationLevel::O3)
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

    if (options.hasArg(OPT_emit_llvm) || lto)
    {
        if (action == Action::ObjectFile || lto)
        {
            // See
            // https://github.com/llvm/llvm-project/blob/ea22fdd120aeb1bbb9ea96670d70193dc02b2c5f/clang/lib/CodeGen/BackendUtil.cpp#L1467
            // Doing full LTO for now
            bool emitLTOSummary = lto && triple.getVendor() != llvm::Triple::Apple;
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
            mpm.addPass(llvm::BitcodeWriterPass(output, false, true));
        }
        else
        {
            mpm.addPass(llvm::PrintModulePass(output));
        }
    }

    mpm.run(*llvmModule, mam);

    if (options.hasArg(OPT_emit_llvm))
    {
        return true;
    }

    llvm::legacy::PassManager codeGenPasses;
    codeGenPasses.add(llvm::createTargetTransformInfoWrapperPass(machine->getTargetIRAnalysis()));
    if (machine->addPassesToEmitFile(codeGenPasses, output, nullptr,
                                     action == Action::Assembly ? llvm::CGFT_AssemblyFile : llvm::CGFT_ObjectFile))
    {
        std::string_view format = action == Action::Assembly ? "Assembly" : "Object file";
        auto arg = options.getLastArg(OPT_target_EQ);
        if (!arg)
        {
            arg = options.getLastArg(OPT_c, OPT_S);
        }
        if (arg)
        {
            llvm::errs() << commandLine
                                .createDiagnosticsBuilder(arg, pylir::Diag::TARGET_N_DOES_NOT_SUPPORT_COMPILING_TO_N,
                                                          triple.str(), format)
                                .addLabel(arg, std::nullopt, pylir::Diag::ERROR_COLOUR)
                                .emitError();
            return false;
        }
        llvm::errs() << pylir::Diag::formatLine(
            pylir::Diag::Error,
            fmt::format(pylir::Diag::TARGET_N_DOES_NOT_SUPPORT_COMPILING_TO_N, triple.str(), format));
        return false;
    }

    codeGenPasses.run(*llvmModule);
    return true;
}

std::unique_ptr<pylir::Toolchain> createToolchainForTriple(const pylir::cli::CommandLine&, llvm::Triple triple)
{
    if (triple.isKnownWindowsMSVCEnvironment())
    {
        return std::make_unique<pylir::MSVCToolchain>(triple);
    }
    if (triple.isOSCygMing())
    {
        return std::make_unique<pylir::MinGWToolchain>(triple);
    }
    if (triple.isOSLinux())
    {
        return std::make_unique<pylir::LinuxToolchain>(triple);
    }
    return {};
}

} // namespace

template <>
struct fmt::formatter<llvm::StringRef> : formatter<string_view>
{
    template <class Context>
    auto format(const llvm::StringRef& string, Context& ctx)
    {
        return fmt::formatter<string_view>::format({string.data(), string.size()}, ctx);
    }
};

int pylir::main(int argc, char* argv[])
{
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeAllAsmParsers();

    pylir::cli::CommandLine commandLine(llvm::sys::fs::getMainExecutable(argv[0], reinterpret_cast<void*>(&main)), argc,
                                        argv);
    if (!commandLine)
    {
        return -1;
    }
    auto& args = commandLine.getArgs();
    if (args.hasArg(OPT_help))
    {
        commandLine.printHelp(llvm::outs());
        return 0;
    }

    if (args.hasArg(OPT_version))
    {
        commandLine.printVersion(llvm::outs());
        return 0;
    }

    auto triple = llvm::Triple::normalize(args.getLastArgValue(OPT_target_EQ, LLVM_DEFAULT_TARGET_TRIPLE));
    auto toolchain = createToolchainForTriple(commandLine, llvm::Triple(triple));
    if (!toolchain)
    {
        auto* arg = args.getLastArg(OPT_target_EQ);
        if (!arg)
        {
            llvm::errs() << pylir::Diag::formatLine(pylir::Diag::Error,
                                                    fmt::format(pylir::Diag::UNSUPPORTED_TARGET_N, triple));
            return -1;
        }
        llvm::errs() << commandLine.createDiagnosticsBuilder(arg, pylir::Diag::UNSUPPORTED_TARGET_N, triple)
                            .addLabel(arg, std::nullopt, pylir::Diag::ERROR_COLOUR)
                            .emitError();
        return -1;
    }

    Action action = Action::Link;
    if (args.hasArg(OPT_fsyntax_only))
    {
        action = Action::SyntaxOnly;
        if (args.hasArg(OPT_S))
        {
            auto assembly = args.getLastArg(OPT_S);
            auto syntaxOnly = args.getLastArg(OPT_fsyntax_only);
            llvm::errs() << commandLine
                                .createDiagnosticsBuilder(
                                    assembly, pylir::Diag::N_WONT_BE_EMITTED_WHEN_ONLY_CHECKING_SYNTAX, "Assembly")
                                .addLabel(assembly, std::nullopt, pylir::Diag::WARNING_COLOUR)
                                .addLabel(syntaxOnly, std::nullopt, pylir::Diag::NOTE_COLOUR)
                                .emitWarning();
        }
        else if (args.hasArg(OPT_c))
        {
            auto objectFile = args.getLastArg(OPT_c);
            auto syntaxOnly = args.getLastArg(OPT_fsyntax_only);
            llvm::errs() << commandLine
                                .createDiagnosticsBuilder(
                                    objectFile, pylir::Diag::N_WONT_BE_EMITTED_WHEN_ONLY_CHECKING_SYNTAX, "Object file")
                                .addLabel(objectFile, std::nullopt, pylir::Diag::WARNING_COLOUR)
                                .addLabel(syntaxOnly, std::nullopt, pylir::Diag::NOTE_COLOUR)
                                .emitWarning();
        }
    }
    else if (args.hasArg(OPT_S))
    {
        action = Action::Assembly;
    }
    else if (args.hasArg(OPT_c))
    {
        action = Action::ObjectFile;
    }

    auto diagExlusiveIR = [&](ID first, std::string_view firstName, ID second, std::string_view secondName)
    {
        if (args.hasArg(second))
        {
            auto lastArg = args.getLastArg(first, second);
            auto secondLast = lastArg->getOption().getID() == first ? args.getLastArg(second) : args.getLastArg(first);
            llvm::errs() << commandLine
                                .createDiagnosticsBuilder(lastArg,
                                                          pylir::Diag::CANNOT_EMIT_N_IR_AND_N_IR_AT_THE_SAME_TIME,
                                                          firstName, secondName)
                                .addLabel(lastArg, std::nullopt, pylir::Diag::ERROR_COLOUR)
                                .addLabel(secondLast, std::nullopt, pylir::Diag::NOTE_COLOUR)
                                .emitError();
            return false;
        }
        return true;
    };

    auto diagActionWithIR = [&](llvm::opt::Arg* arg, std::string_view name)
    {
        if (args.hasArg(OPT_fsyntax_only))
        {
            auto syntaxOnly = args.getLastArg(OPT_fsyntax_only);
            llvm::errs() << commandLine
                                .createDiagnosticsBuilder(
                                    arg, pylir::Diag::N_IR_WONT_BE_EMITTED_WHEN_ONLY_CHECKING_SYNTAX, name)
                                .addLabel(arg, std::nullopt, pylir::Diag::WARNING_COLOUR)
                                .addLabel(syntaxOnly, std::nullopt, pylir::Diag::NOTE_COLOUR)
                                .emitWarning();
        }
        if (action == Action::Link)
        {
            llvm::errs() << commandLine.createDiagnosticsBuilder(arg, pylir::Diag::CANNOT_EMIT_N_IR_WHEN_LINKING, name)
                                .addLabel(arg, std::nullopt, pylir::Diag::ERROR_COLOUR)
                                .emitError();
            return false;
        }
        return true;
    };

    if (auto* emitLLVM = args.getLastArg(OPT_emit_llvm))
    {
        if (!diagExlusiveIR(OPT_emit_llvm, "LLVM", OPT_emit_mlir, "MLIR")
            || !diagExlusiveIR(OPT_emit_llvm, "LLVM", OPT_emit_pylir, "Pylir"))
        {
            return -1;
        }
        if (!diagActionWithIR(emitLLVM, "LLVM"))
        {
            return -1;
        }
    }
    else if (auto* emitMLIR = args.getLastArg(OPT_emit_mlir))
    {
        if (!diagExlusiveIR(OPT_emit_mlir, "MLIR", OPT_emit_pylir, "Pylir"))
        {
            return -1;
        }
        if (!diagActionWithIR(emitMLIR, "MLIR"))
        {
            return -1;
        }
    }
    else if (auto* emitPylir = args.getLastArg(OPT_emit_pylir))
    {
        if (!diagActionWithIR(emitPylir, "Pylir"))
        {
            return -1;
        }
    }

    if (args.hasMultipleArgs(OPT_INPUT))
    {
        auto* second = *std::next(args.filtered(OPT_INPUT).begin());
        llvm::errs() << commandLine.createDiagnosticsBuilder(second, pylir::Diag::EXPECTED_ONLY_ONE_INPUT_FILE)
                            .addLabel(second, std::nullopt, pylir::Diag::ERROR_COLOUR)
                            .emitError();
        return -1;
    }

    auto* inputFile = args.getLastArg(OPT_INPUT);
    if (!inputFile)
    {
        llvm::errs() << pylir::Diag::formatLine(Diag::Error, fmt::format(pylir::Diag::NO_INPUT_FILE));
        return -1;
    }

    auto fd = llvm::sys::fs::openNativeFileForRead(inputFile->getValue());
    if (!fd)
    {
        llvm::consumeError(fd.takeError());
        llvm::errs() << commandLine
                            .createDiagnosticsBuilder(inputFile, pylir::Diag::FAILED_TO_OPEN_FILE_N,
                                                      inputFile->getValue())
                            .addLabel(inputFile, std::nullopt, pylir::Diag::ERROR_COLOUR)
                            .emitError();
        return -1;
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
            return -1;
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
        return -1;
    }
    exit.reset();

    pylir::Diag::Document doc(std::move(content), inputFile->getValue());
    pylir::Parser parser(doc);
    auto tree = parser.parseFileInput();
    if (!tree)
    {
        llvm::errs() << tree.error();
        return -1;
    }
    if (args.hasArg(OPT_emit_ast))
    {
        pylir::Dumper dumper;
        llvm::outs() << dumper.dump(*tree);
    }
    if (action == Action::SyntaxOnly)
    {
        return 0;
    }

    auto filename = llvm::sys::path::filename(inputFile->getValue()).str();
    llvm::SmallString<20> realOutputFilename;
    if (action == Action::Link)
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
            if (action == Action::ObjectFile)
            {
                defaultName = filename + ".bc";
            }
            else
            {
                defaultName = filename + ".ll";
            }
        }
        else if (action == Action::Assembly)
        {
            defaultName = filename + ".s";
        }
        else if (action == Action::ObjectFile)
        {
            defaultName = filename + ".o";
        }
        realOutputFilename = args.getLastArgValue(OPT_o, defaultName);
    }

    std::optional<llvm::sys::fs::TempFile> outputFile;
    llvm::SmallString<20> tempFileName = realOutputFilename;
    if (realOutputFilename != "-")
    {
        auto extension = llvm::sys::path::extension(realOutputFilename);
        llvm::sys::path::remove_filename(tempFileName);
        llvm::sys::path::append(tempFileName, llvm::sys::path::stem(realOutputFilename) + "-%%%%" + extension);
        auto tempFile = llvm::sys::fs::TempFile::create(tempFileName);
        if (!tempFile)
        {
            llvm::consumeError(tempFile.takeError());
            llvm::errs() << pylir::Diag::formatLine(
                pylir::Diag::Error, fmt::format(pylir::Diag::FAILED_TO_CREATE_TEMPORARY_FILE_N, tempFileName.str()));
            return -1;
        }
        outputFile = std::move(tempFile.get());
    }

    bool success;
    {
        if (!outputFile)
        {
            success = executeCompilation(action, *tree, doc, commandLine, llvm::outs());
        }
        else
        {
            auto output = llvm::raw_fd_ostream(outputFile->FD, false);
            success = executeCompilation(action, *tree, doc, commandLine, output);
        }
    }
    if (!success)
    {
        if (outputFile)
        {
            if (auto error = outputFile->discard())
            {
                llvm::consumeError(std::move(error));
                llvm::errs() << pylir::Diag::formatLine(
                    pylir::Diag::Error,
                    fmt::format(pylir::Diag::FAILED_TO_DISCARD_TEMPORARY_FILE_N, tempFileName.str()));
                return -1;
            }
        }
        return -1;
    }
    if (action != Action::Link)
    {
        if (outputFile)
        {
            if (auto error = outputFile->keep(realOutputFilename))
            {
                llvm::errs() << pylir::Diag::formatLine(pylir::Diag::Error,
                                                        fmt::format(pylir::Diag::FAILED_TO_RENAME_TEMPORARY_FILE_N_TO_N,
                                                                    tempFileName.str(), realOutputFilename.str()));
                return -1;
            }
        }
        return 0;
    }
    success = toolchain->link(commandLine, outputFile->TmpName);
    if (auto error = outputFile->discard())
    {
        llvm::consumeError(std::move(error));
        llvm::errs() << pylir::Diag::formatLine(
            pylir::Diag::Error, fmt::format(pylir::Diag::FAILED_TO_DISCARD_TEMPORARY_FILE_N, tempFileName.str()));
        return -1;
    }
    return success ? 0 : -1;
}
