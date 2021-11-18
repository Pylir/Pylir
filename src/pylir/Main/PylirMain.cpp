
#include "PylirMain.hpp"

#include <mlir/Dialect/LLVMIR/Transforms/LegalizeForExport.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

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

namespace
{
enum ID
{
    OPT_INVALID = 0, // This is not an option ID.
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM, HELPTEXT, METAVAR, VALUES) OPT_##ID,
#include <pylir/Main/Opts.inc>
#undef OPTION
};

#define PREFIX(NAME, VALUE) const char* const NAME[] = VALUE;
#include <pylir/Main/Opts.inc>
#undef PREFIX

static const llvm::opt::OptTable::Info InfoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM, HELPTEXT, METAVAR, VALUES) \
    {PREFIX, NAME,  HELPTEXT,    METAVAR,     OPT_##ID,  llvm::opt::Option::KIND##Class,                 \
     PARAM,  FLAGS, OPT_##GROUP, OPT_##ALIAS, ALIASARGS, VALUES},
#include <pylir/Main/Opts.inc>
#undef OPTION
};

enum class Action
{
    SyntaxOnly,
    ObjectFile,
    Assembly
};

} // namespace

namespace pylir::Diag
{
template <>
struct LocationProvider<llvm::opt::Arg*, void>
{
    static std::pair<std::size_t, std::size_t> getRange(llvm::opt::Arg* value, const void* context) noexcept
    {
        auto* commandLine = reinterpret_cast<const pylir::cli::CommandLine*>(context);
        return commandLine->m_argRanges.lookup(value);
    }
};
} // namespace pylir::Diag

namespace
{
bool executeAction(Action action, pylir::Diag::Document& file, const pylir::cli::CommandLine& commandLine)
{
    auto& options = commandLine.getArgs();
    pylir::Parser parser(file);
    auto tree = parser.parseFileInput();
    if (!tree)
    {
        llvm::errs() << tree.error();
        return false;
    }
    if (options.hasArg(OPT_emit_ast))
    {
        pylir::Dumper dumper;
        llvm::outs() << dumper.dump(*tree);
    }
    if (action == Action::SyntaxOnly)
    {
        return true;
    }
    mlir::MLIRContext context;
    context.getDiagEngine().registerHandler([](mlir::Diagnostic& diagnostic) { diagnostic.print(llvm::errs()); });
    auto module = pylir::codegen(&context, *tree, file);

    auto filename = llvm::sys::path::filename(file.getFilename()).str();

    std::string defaultName;
    if (options.hasArg(OPT_emit_mlir))
    {
        defaultName = filename + ".mlir";
    }
    else if (options.hasArg(OPT_emit_llvm))
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
    else
    {
        if (action == Action::ObjectFile)
        {
            defaultName = filename + ".o";
        }
        else
        {
            defaultName = filename + ".s";
        }
    }

    std::error_code errorCode;
    auto outputString = options.getLastArgValue(OPT_o, defaultName);
    auto output = llvm::raw_fd_ostream(outputString, errorCode);
    if (errorCode)
    {
        auto outputArg = options.getLastArg(OPT_o);
        if (outputArg)
        {
            llvm::errs() << commandLine
                                .createDiagnosticsBuilder(outputArg, pylir::Diag::FAILED_TO_OPEN_FILE_N, outputString)
                                .addLabel(outputArg, std::nullopt, pylir::Diag::ERROR_COLOUR)
                                .emitError();
        }
        else
        {
            llvm::errs() << pylir::Diag::formatLine(pylir::Diag::Error,
                                                    fmt::format(pylir::Diag::FAILED_TO_OPEN_FILE_N, outputString));
        }
        return false;
    }
    mlir::PassManager manager(&context);
#ifndef NDEBUG
    manager.enableVerifier();
    #if !defined(__MINGW32_MAJOR_VERSION) || !defined(__clang__)
    manager.enableCrashReproducerGeneration("failure.mlir");
    #endif
    manager.enableIRPrinting(std::make_unique<mlir::PassManager::IRPrinterConfig>(false, false, true));
#endif
    // TODO add transformations
    if (options.hasArg(OPT_emit_mlir))
    {
        if (mlir::failed(manager.run(*module)))
        {
            return false;
        }
        module->print(output, mlir::OpPrintingFlags{}.enableDebugInfo());
        return true;
    }
    manager.addNestedPass<mlir::FuncOp>(pylir::Py::createExpandPyDialectPass());

    auto triple = llvm::Triple(options.getLastArgValue(OPT_target, LLVM_DEFAULT_TARGET_TRIPLE));
    std::string error;
    auto* targetM = llvm::TargetRegistry::lookupTarget(triple.str(), error);
    if (!targetM)
    {
        auto outputArg = options.getLastArg(OPT_target);
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
        mpm = passBuilder.buildPerModuleDefaultPipeline(level);
    }

    if (options.hasArg(OPT_emit_llvm))
    {
        if (action == Action::ObjectFile)
        {
            mpm.addPass(llvm::BitcodeWriterPass(output));
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
                                     action == Action::ObjectFile ? llvm::CGFT_ObjectFile : llvm::CGFT_AssemblyFile))
    {
        if (action == Action::Assembly)
        {
            auto arg = options.getLastArg(OPT_S);
            llvm::errs() << commandLine
                                .createDiagnosticsBuilder(arg, pylir::Diag::TARGET_N_DOES_NOT_SUPPORT_COMPILING_TO_N,
                                                          triple.str(), "Assembly")
                                .addLabel(arg, std::nullopt, pylir::Diag::ERROR_COLOUR)
                                .emitError();
            return false;
        }
        llvm::errs() << pylir::Diag::formatLine(
            pylir::Diag::Error,
            fmt::format(pylir::Diag::TARGET_N_DOES_NOT_SUPPORT_COMPILING_TO_N, triple.str(), "Object file"));
        return false;
    }

    codeGenPasses.run(*llvmModule);

    return true;
}

} // namespace

pylir::cli::PylirOptTable::PylirOptTable() : llvm::opt::OptTable(InfoTable) {}

pylir::cli::CommandLine::CommandLine(int argc, char** argv)
    : m_saver(m_allocator),
      m_args(m_table.parseArgs(argc, argv, OPT_UNKNOWN, m_saver,
                               [this](llvm::StringRef msg)
                               {
                                   m_errorsOccurred = true;
                                   llvm::errs() << pylir::Diag::formatLine(pylir::Diag::Severity::Error, msg);
                               })),
      m_exe(argv[0]),
      m_rendered(
          [this]
          {
              std::string rendered = llvm::sys::path::filename(m_exe).str();
              for (auto iter : m_args)
              {
                  auto arg = iter->getAsString(m_args);
                  rendered += " ";
                  m_argRanges.insert({iter, {rendered.size(), rendered.size() + arg.size()}});
                  rendered += arg;
              }
              return rendered;
          }(),
          "<command-line>")
{
}

void pylir::cli::CommandLine::printHelp(llvm::raw_ostream& out) const
{
    m_table.printHelp(out, (llvm::sys::path::filename(m_exe) + " [options] <input>").str().c_str(),
                      "Python optimizing MLIR compiler");
}

void pylir::cli::CommandLine::printVersion(llvm::raw_ostream& out) const
{
    out << "pylir " PYLIR_VERSION "\n";
}

int pylir::main(int argc, char* argv[])
{
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeAllAsmParsers();

    pylir::cli::CommandLine commandLine(argc, argv);
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

    if (args.hasArg(OPT_emit_llvm))
    {
        if (args.hasArg(OPT_emit_mlir))
        {
            auto lastArg = args.getLastArg(OPT_emit_llvm, OPT_emit_mlir);
            auto secondLast = lastArg->getOption().getID() == OPT_emit_llvm ? args.getLastArg(OPT_emit_mlir) :
                                                                              args.getLastArg(OPT_emit_llvm);
            llvm::errs() << commandLine
                                .createDiagnosticsBuilder(lastArg,
                                                          pylir::Diag::CANNOT_EMIT_LLVM_IR_AND_MLIR_IR_AT_THE_SAME_TIME)
                                .addLabel(lastArg, std::nullopt, pylir::Diag::ERROR_COLOUR)
                                .addLabel(secondLast, std::nullopt, pylir::Diag::NOTE_COLOUR)
                                .emitError();
            return -1;
        }
        if (args.hasArg(OPT_fsyntax_only))
        {
            auto llvmArg = args.getLastArg(OPT_emit_llvm);
            auto syntaxOnly = args.getLastArg(OPT_fsyntax_only);
            llvm::errs() << commandLine
                                .createDiagnosticsBuilder(
                                    llvmArg, pylir::Diag::LLVM_IR_WONT_BE_EMITTED_WHEN_ONLY_CHECKING_SYNTAX)
                                .addLabel(llvmArg, std::nullopt, pylir::Diag::WARNING_COLOUR)
                                .addLabel(syntaxOnly, std::nullopt, pylir::Diag::NOTE_COLOUR)
                                .emitWarning();
        }
    }
    else if (args.hasArg(OPT_emit_mlir))
    {
        if (args.hasArg(OPT_fsyntax_only))
        {
            auto mlirArg = args.getLastArg(OPT_emit_mlir);
            auto syntaxOnly = args.getLastArg(OPT_fsyntax_only);
            llvm::errs() << commandLine
                                .createDiagnosticsBuilder(
                                    mlirArg, pylir::Diag::MLIR_IR_WONT_BE_EMITTED_WHEN_ONLY_CHECKING_SYNTAX)
                                .addLabel(mlirArg, std::nullopt, pylir::Diag::WARNING_COLOUR)
                                .addLabel(syntaxOnly, std::nullopt, pylir::Diag::NOTE_COLOUR)
                                .emitWarning();
        }
    }

    Action action = Action::ObjectFile;
    if (args.hasArg(OPT_fsyntax_only))
    {
        action = Action::SyntaxOnly;
        if (args.hasArg(OPT_S))
        {
            auto assembly = args.getLastArg(OPT_S);
            auto syntaxOnly = args.getLastArg(OPT_fsyntax_only);
            llvm::errs() << commandLine
                                .createDiagnosticsBuilder(
                                    assembly, pylir::Diag::ASSEMBLY_WONT_BE_EMITTED_WHEN_ONLY_CHECKING_SYNTAX)
                                .addLabel(assembly, std::nullopt, pylir::Diag::WARNING_COLOUR)
                                .addLabel(syntaxOnly, std::nullopt, pylir::Diag::NOTE_COLOUR)
                                .emitWarning();
        }
    }
    else if (args.hasArg(OPT_S))
    {
        action = Action::Assembly;
    }

    for (auto& iter : args.getAllArgValues(OPT_c))
    {
        pylir::Diag::Document doc(iter);
        if (!executeAction(action, doc, commandLine))
        {
            return -1;
        }
    }

    for (auto& iter : args.filtered(OPT_INPUT))
    {
        auto fd = llvm::sys::fs::openNativeFileForRead(iter->getValue());
        if (!fd)
        {
            llvm::consumeError(fd.takeError());
            llvm::errs() << commandLine
                                .createDiagnosticsBuilder(iter, pylir::Diag::FAILED_TO_OPEN_FILE_N, iter->getValue())
                                .addLabel(iter, std::nullopt, pylir::Diag::ERROR_COLOUR)
                                .emitError();
            return -1;
        }
        auto exit = llvm::make_scope_exit([&fd] { llvm::sys::fs::closeFile(*fd); });
        llvm::sys::fs::file_status status;
        auto error = llvm::sys::fs::status(*fd, status);
        if (error)
        {
            llvm::errs() << commandLine
                                .createDiagnosticsBuilder(iter, pylir::Diag::FAILED_TO_ACCESS_FILE_N, iter->getValue())
                                .addLabel(iter, std::nullopt, pylir::Diag::ERROR_COLOUR)
                                .emitError();
            return -1;
        }
        std::string content(status.getSize(), '\0');
        auto read = llvm::sys::fs::readNativeFile(*fd, {content.data(), content.size()});
        if (!read)
        {
            llvm::consumeError(fd.takeError());
            llvm::errs() << commandLine
                                .createDiagnosticsBuilder(iter, pylir::Diag::FAILED_TO_READ_FILE_N, iter->getValue())
                                .addLabel(iter, std::nullopt, pylir::Diag::ERROR_COLOUR)
                                .emitError();
            return -1;
        }

        pylir::Diag::Document doc(std::move(content), iter->getValue());
        if (!executeAction(action, doc, commandLine))
        {
            return -1;
        }
    }

    return 0;
}
