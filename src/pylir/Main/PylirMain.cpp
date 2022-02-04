
#include "PylirMain.hpp"

#include <llvm/Option/Arg.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/TargetSelect.h>

#include <pylir/Diagnostics/DiagnosticMessages.hpp>
#include <pylir/Main/Opts.inc>

#include "CommandLine.hpp"
#include "CompilerInvocation.hpp"
#include "LinuxToolchain.hpp"
#include "MSVCToolchain.hpp"
#include "MinGWToolchain.hpp"
#include "Toolchain.hpp"

using namespace pylir::cli;

namespace
{

std::unique_ptr<pylir::Toolchain> createToolchainForTriple(const pylir::cli::CommandLine& commandLine,
                                                           const llvm::Triple& triple)
{
    if (triple.isKnownWindowsMSVCEnvironment())
    {
        return std::make_unique<pylir::MSVCToolchain>(triple, commandLine);
    }
    if (triple.isOSCygMing())
    {
        return std::make_unique<pylir::MinGWToolchain>(triple, commandLine);
    }
    if (triple.isOSLinux())
    {
        return std::make_unique<pylir::LinuxToolchain>(triple, commandLine);
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

int pylir::main(int argc, char** argv)
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
    const auto& args = commandLine.getArgs();
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

    pylir::CompilerInvocation::Action action = pylir::CompilerInvocation::Link;
    if (args.hasArg(OPT_fsyntax_only))
    {
        action = pylir::CompilerInvocation::SyntaxOnly;
        if (args.hasArg(OPT_S))
        {
            auto *assembly = args.getLastArg(OPT_S);
            auto *syntaxOnly = args.getLastArg(OPT_fsyntax_only);
            llvm::errs() << commandLine
                                .createDiagnosticsBuilder(
                                    assembly, pylir::Diag::N_WONT_BE_EMITTED_WHEN_ONLY_CHECKING_SYNTAX, "Assembly")
                                .addLabel(assembly, std::nullopt, pylir::Diag::WARNING_COLOUR)
                                .addLabel(syntaxOnly, std::nullopt, pylir::Diag::NOTE_COLOUR)
                                .emitWarning();
        }
        else if (args.hasArg(OPT_c))
        {
            auto *objectFile = args.getLastArg(OPT_c);
            auto *syntaxOnly = args.getLastArg(OPT_fsyntax_only);
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
        action = pylir::CompilerInvocation::Assembly;
    }
    else if (args.hasArg(OPT_c))
    {
        action = pylir::CompilerInvocation::ObjectFile;
    }

    auto diagExlusiveIR = [&](ID first, std::string_view firstName, ID second, std::string_view secondName)
    {
        if (args.hasArg(second))
        {
            auto *lastArg = args.getLastArg(first, second);
            auto *secondLast = lastArg->getOption().getID() == first ? args.getLastArg(second) : args.getLastArg(first);
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
            auto *syntaxOnly = args.getLastArg(OPT_fsyntax_only);
            llvm::errs() << commandLine
                                .createDiagnosticsBuilder(
                                    arg, pylir::Diag::N_IR_WONT_BE_EMITTED_WHEN_ONLY_CHECKING_SYNTAX, name)
                                .addLabel(arg, std::nullopt, pylir::Diag::WARNING_COLOUR)
                                .addLabel(syntaxOnly, std::nullopt, pylir::Diag::NOTE_COLOUR)
                                .emitWarning();
        }
        if (action == pylir::CompilerInvocation::Link)
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

    if (auto *opt = args.getLastArg(OPT_O);
        opt && opt->getValue() == std::string_view{"4"} && !args.hasArg(OPT_emit_mlir, OPT_emit_pylir, OPT_emit_llvm)
        && (action == pylir::CompilerInvocation::Assembly || action == pylir::CompilerInvocation::ObjectFile)
        && !args.hasArg(OPT_flto, OPT_fno_lto))
    {
        llvm::errs() << commandLine
                            .createDiagnosticsBuilder(opt, pylir::Diag::O4_MAY_ENABLE_LTO_COMPILER_MIGHT_OUTPUT_LLVM_IR,
                                                      action == pylir::CompilerInvocation::Assembly ? "Assembly file" :
                                                                                                      "Object file")
                            .addLabel(opt, std::nullopt, pylir::Diag::WARNING_COLOUR)
                            .emitWarning();
    }

    if (auto *opt = args.getLastArg(OPT_flto, OPT_fno_lto);
        opt && opt->getOption().matches(OPT_flto) && !args.hasArg(OPT_emit_mlir, OPT_emit_pylir, OPT_emit_llvm)
        && (action == pylir::CompilerInvocation::Assembly || action == pylir::CompilerInvocation::ObjectFile))
    {
        llvm::errs() << commandLine
                            .createDiagnosticsBuilder(opt, pylir::Diag::LTO_ENABLED_COMPILER_WILL_OUTPUT_LLVM_IR,
                                                      action == pylir::CompilerInvocation::Assembly ? "Assembly file" :
                                                                                                      "Object file")
                            .addLabel(opt, std::nullopt, pylir::Diag::WARNING_COLOUR)
                            .emitWarning();
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

    pylir::CompilerInvocation invocation;
    return mlir::succeeded(invocation.executeAction(inputFile, commandLine, *toolchain, action)) ? 0 : -1;
}
