
#include "PylirMain.hpp"

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
#include <llvm/Option/Arg.h>
#include <llvm/Option/ArgList.h>
#include <llvm/Option/Option.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/StringSaver.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>

#include <pylir/CodeGen/CodeGen.hpp>
#include <pylir/Diagnostics/DiagnosticMessages.hpp>
#include <pylir/Diagnostics/Document.hpp>
#include <pylir/Main/Opts.inc>
#include <pylir/Optimizer/Conversion/PylirToLLVM.hpp>
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

class PylirOptTable : public llvm::opt::OptTable
{
public:
    PylirOptTable() : OptTable(InfoTable) {}
};

enum class Action
{
    SyntaxOnly,
    ObjectFile,
    Assembly
};

std::optional<pylir::Diag::Document>& getCLIDoc()
{
    thread_local static std::optional<pylir::Diag::Document> document;
    return document;
}

llvm::DenseMap<llvm::opt::Arg*, std::pair<std::size_t, std::size_t>>& getArgRanges()
{
    thread_local static llvm::DenseMap<llvm::opt::Arg*, std::pair<std::size_t, std::size_t>> ranges;
    return ranges;
}

} // namespace

namespace pylir::Diag
{
template <>
struct LocationProvider<llvm::opt::Arg*, void>
{
    static std::pair<std::size_t, std::size_t> getRange(llvm::opt::Arg* value) noexcept
    {
        return getArgRanges().lookup(value);
    }
};
} // namespace pylir::Diag

namespace
{
template <class T, class S, class... Args>
[[nodiscard]] pylir::Diag::DiagnosticsBuilder createDiagnosticsBuilder(const T& location, const S& message,
                                                                       Args&&... args)
{
    return pylir::Diag::DiagnosticsBuilder(*getCLIDoc(), location, message, std::forward<Args>(args)...);
}

bool executeAction(Action action, pylir::Diag::Document& file, llvm::opt::InputArgList& options)
{
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
    auto triple = llvm::Triple(options.getLastArgValue(OPT_target, LLVM_DEFAULT_TARGET_TRIPLE));
    mlir::MLIRContext context;
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
            llvm::errs() << createDiagnosticsBuilder(outputArg, pylir::Diag::FAILED_TO_OPEN_FILE_N, outputString)
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
    // TODO add transformations
    if (options.hasArg(OPT_emit_mlir))
    {
        if (mlir::failed(manager.run(*module)))
        {
            return -1;
        }
        module->print(output, mlir::OpPrintingFlags{}.enableDebugInfo());
        return true;
    }
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
        llvm::errs() << createDiagnosticsBuilder(outputArg, pylir::Diag::COULD_NOT_FIND_TARGET_N, triple.str())
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
        llvm::errs() << createDiagnosticsBuilder(optArg, pylir::Diag::INVALID_OPTIMIZATION_LEVEL_N,
                                                 optArg->getAsString(options))
                            .addLabel(optArg, std::nullopt, pylir::Diag::ERROR_COLOUR)
                            .emitError();
        return false;
    }

    auto machine = std::unique_ptr<llvm::TargetMachine>(
        targetM->createTargetMachine(triple.str(), "generic", "", {}, llvm::Reloc::Static, {}, *optLevel));

    std::string passOptions =
        "target-triple=" + triple.str() + " data-layout=" + machine->createDataLayout().getStringRepresentation();

    auto pass = pylir::Dialect::createConvertPylirToLLVMPass();
    if (mlir::failed(pass->initializeOptions(passOptions)))
    {
        return false;
    }

    manager.addPass(std::move(pass));
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
        mpm = passBuilder.buildO0DefaultPipeline(llvm::PassBuilder::OptimizationLevel::O0);
    }
    else
    {
        llvm::PassBuilder::OptimizationLevel level =
            llvm::StringSwitch<llvm::PassBuilder::OptimizationLevel>(options.getLastArgValue(OPT_O))
                .Case("1", llvm::PassBuilder::OptimizationLevel::O1)
                .Case("2", llvm::PassBuilder::OptimizationLevel::O2)
                .Case("3", llvm::PassBuilder::OptimizationLevel::O3)
                .Case("s", llvm::PassBuilder::OptimizationLevel::Os)
                .Case("z", llvm::PassBuilder::OptimizationLevel::Oz);
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
            llvm::errs() << createDiagnosticsBuilder(arg, pylir::Diag::TARGET_N_DOES_NOT_SUPPORT_COMPILING_TO_N,
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

int pylir::main(int argc, char* argv[])
{
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeAllAsmParsers();

    llvm::BumpPtrAllocator a;
    llvm::StringSaver saver(a);
    PylirOptTable table;
    bool errorOccured = false;
    auto args = table.parseArgs(argc, argv, OPT_UNKNOWN, saver,
                                [&](llvm::StringRef msg)
                                {
                                    errorOccured = true;
                                    llvm::errs() << pylir::Diag::formatLine(pylir::Diag::Severity::Error, msg);
                                });
    if (errorOccured)
    {
        return -1;
    }
    if (args.hasArg(OPT_help))
    {
        table.printHelp(llvm::outs(), (llvm::Twine(argv[0]) + " [options] <input>").str().c_str(),
                        "Python optimizing MLIR compiler");
        return 0;
    }

    if (args.hasArg(OPT_version))
    {
        llvm::outs() << "pylir " PYLIR_VERSION "\n";
        return 0;
    }

    std::string rendered = llvm::sys::path::filename(argv[0]).str();
    for (auto iter : args)
    {
        auto arg = iter->getAsString(args);
        rendered += " ";
        getArgRanges().insert({iter, {rendered.size(), rendered.size() + arg.size()}});
        rendered += arg;
    }
    getCLIDoc().emplace(rendered, "<command-line>");

    if (args.hasArg(OPT_emit_llvm))
    {
        if (args.hasArg(OPT_emit_mlir))
        {
            auto lastArg = args.getLastArg(OPT_emit_llvm, OPT_emit_mlir);
            auto secondLast = lastArg->getOption().getID() == OPT_emit_llvm ? args.getLastArg(OPT_emit_mlir) :
                                                                              args.getLastArg(OPT_emit_llvm);
            llvm::errs() << createDiagnosticsBuilder(lastArg,
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
            llvm::errs() << createDiagnosticsBuilder(llvmArg,
                                                     pylir::Diag::LLVM_IR_WONT_BE_EMITTED_WHEN_ONLY_CHECKING_SYNTAX)
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
            llvm::errs() << createDiagnosticsBuilder(mlirArg,
                                                     pylir::Diag::MLIR_IR_WONT_BE_EMITTED_WHEN_ONLY_CHECKING_SYNTAX)
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
            llvm::errs() << createDiagnosticsBuilder(assembly,
                                                     pylir::Diag::ASSEMBLY_WONT_BE_EMITTED_WHEN_ONLY_CHECKING_SYNTAX)
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
        if (!executeAction(action, doc, args))
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
            llvm::errs() << createDiagnosticsBuilder(iter, pylir::Diag::FAILED_TO_OPEN_FILE_N, iter->getValue())
                                .addLabel(iter, std::nullopt, pylir::Diag::ERROR_COLOUR)
                                .emitError();
            return -1;
        }
        auto exit = llvm::make_scope_exit([&fd] { llvm::sys::fs::closeFile(*fd); });
        llvm::sys::fs::file_status status;
        auto error = llvm::sys::fs::status(*fd, status);
        if (error)
        {
            llvm::errs() << createDiagnosticsBuilder(iter, pylir::Diag::FAILED_TO_ACCESS_FILE_N, iter->getValue())
                                .addLabel(iter, std::nullopt, pylir::Diag::ERROR_COLOUR)
                                .emitError();
            return -1;
        }
        std::string content(status.getSize(), '\0');
        auto read = llvm::sys::fs::readNativeFile(*fd, {content.data(), content.size()});
        if (!read)
        {
            llvm::consumeError(fd.takeError());
            llvm::errs() << createDiagnosticsBuilder(iter, pylir::Diag::FAILED_TO_READ_FILE_N, iter->getValue())
                                .addLabel(iter, std::nullopt, pylir::Diag::ERROR_COLOUR)
                                .emitError();
            return -1;
        }

        pylir::Diag::Document doc(std::move(content), iter->getValue());
        if (!executeAction(action, doc, args))
        {
            return -1;
        }
    }

    return 0;
}
