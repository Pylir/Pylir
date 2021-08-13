
#include "PylirMain.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/Triple.h>
#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/Option/Arg.h>
#include <llvm/Option/ArgList.h>
#include <llvm/Option/Option.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/StringSaver.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>

#include <pylir/CodeGen/CodeGen.hpp>
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
    auto triple = llvm::Triple(LLVM_DEFAULT_TARGET_TRIPLE);
    mlir::MLIRContext context;
    auto module = pylir::codegen(&context, *tree, file);

    std::string defaultName = "a";
    if (options.hasArg(OPT_emit_mlir))
    {
        defaultName += ".mlir";
    }
    else if (options.hasArg(OPT_emit_llvm))
    {
        if (action == Action::ObjectFile)
        {
            defaultName += ".bc";
        }
        else
        {
            defaultName += ".ll";
        }
    }
    else
    {
        if (triple.isOSWindows())
        {
            defaultName += ".exe";
        }
        else
        {
            defaultName += ".out";
        }
    }

    std::error_code errorCode;
    auto output = llvm::raw_fd_ostream(options.getLastArgValue(OPT_o, defaultName), errorCode);
    if (!errorCode)
    {
        // TODO error could not open output file
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
        // TODO error did not find target;
        return false;
    }
    auto machine = std::unique_ptr<llvm::TargetMachine>(targetM->createTargetMachine(
        triple.str(), "generic", "", {}, llvm::Reloc::Static, {}, llvm::CodeGenOpt::Aggressive));

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

    //TODO adjust opt level
    auto mpm = passBuilder.buildPerModuleDefaultPipeline(llvm::PassBuilder::OptimizationLevel::O3);
    mpm.run(*llvmModule, mam);

    //TODO: append legacy pass manager for codegen

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
    auto args = table.parseArgs(argc, argv, OPT_UNKNOWN, saver,
                                [&](llvm::StringRef msg)
                                {
                                    // TODO:
                                    llvm::errs() << msg;
                                    std::terminate();
                                });

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

    if (args.hasArg(OPT_emit_llvm))
    {
        if (args.hasArg(OPT_emit_mlir))
        {
            // TODO error
            return -1;
        }
        if (args.hasArg(OPT_fsyntax_only))
        {
            // TODO warning
        }
    }
    else if (args.hasArg(OPT_emit_mlir))
    {
        if (args.hasArg(OPT_fsyntax_only))
        {
            // TODO warning
        }
    }

    Action action = Action::ObjectFile;
    if (args.hasArg(OPT_fsyntax_only))
    {
        action = Action::SyntaxOnly;
        if (args.hasArg(OPT_S))
        {
            // TODO warning
        }
    }
    else if (args.hasArg(OPT_S))
    {
        action = Action::Assembly;
    }

    if (action == Action::ObjectFile && args.hasArg(OPT_emit_mlir))
    {
        // TODO error unsupported
        return -1;
    }

    for (auto& iter : args.getAllArgValues(OPT_c))
    {
        pylir::Diag::Document doc(iter);
        if (!executeAction(action, doc, args))
        {
            return -1;
        }
    }

    for (auto& iter : args.getAllArgValues(OPT_INPUT))
    {
        auto fd = llvm::sys::fs::openNativeFileForRead(iter);
        if (!fd)
        {
            // TODO error could not open file
            return -1;
        }
        auto exit = llvm::make_scope_exit([&fd] { llvm::sys::fs::closeFile(*fd); });
        llvm::sys::fs::file_status status;
        auto error = llvm::sys::fs::status(*fd, status);
        if (!error)
        {
            // TODO error accessing
            return -1;
        }
        llvm::sys::fs::mapped_file_region mapped(*fd, llvm::sys::fs::mapped_file_region::readonly, status.getSize(), 0,
                                                 error);
        if (!error)
        {
            // TODO error mapping
            return -1;
        }
        pylir::Diag::Document doc(std::string{mapped.const_data(), mapped.size()}, iter);
        if (!executeAction(action, doc, args))
        {
            return -1;
        }
    }

    return 0;
}
