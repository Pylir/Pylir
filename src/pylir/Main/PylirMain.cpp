//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PylirMain.hpp"

#include <mlir/Transforms/Passes.h>

#include <llvm/Option/Arg.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/TargetSelect.h>

#include <pylir/Diagnostics/DiagnosticsManager.hpp>
#include <pylir/Main/Opts.inc>
#include <pylir/Optimizer/Optimizer.hpp>
#include <pylir/Optimizer/PylirPy/Transforms/Passes.hpp>
#include <pylir/Optimizer/Transforms/Passes.hpp>

#include "CommandLine.hpp"
#include "CompilerInvocation.hpp"
#include "DarwinToolchain.hpp"
#include "DiagnosticMessages.hpp"
#include "DiagnosticsVerifier.hpp"
#include "LinuxToolchain.hpp"
#include "MSVCToolchain.hpp"
#include "MinGWToolchain.hpp"
#include "Toolchain.hpp"

using namespace pylir::cli;

namespace {

std::unique_ptr<pylir::Toolchain>
createToolchainForTriple(pylir::cli::CommandLine& commandLine,
                         const llvm::Triple& triple) {
  if (triple.isKnownWindowsMSVCEnvironment())
    return std::make_unique<pylir::MSVCToolchain>(triple, commandLine);

  if (triple.isOSCygMing())
    return std::make_unique<pylir::MinGWToolchain>(triple, commandLine);

  if (triple.isOSLinux())
    return std::make_unique<pylir::LinuxToolchain>(triple, commandLine);

  if (triple.isOSDarwin())
    return std::make_unique<pylir::DarwinToolchain>(triple, commandLine);

  return {};
}

} // namespace

int pylir::main(int argc, char** argv) {
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();
  mlir::registerCSEPass();
  mlir::registerCanonicalizerPass();
  mlir::registerSCCPPass();
  mlir::registerSymbolDCEPass();
  pylir::registerTransformPasses();
  pylir::Py::registerTransformPasses();
  pylir::registerOptimizationPipelines();

  pylir::Diag::DiagnosticsManager diagnosticManager;
  pylir::cli::CommandLine commandLine(
      llvm::sys::fs::getMainExecutable(argv[0], reinterpret_cast<void*>(&main)),
      argc, argv, diagnosticManager);
  if (!commandLine)
    return -1;

  const auto& args = commandLine.getArgs();
  if (args.hasArg(OPT_help)) {
    commandLine.printHelp(llvm::outs());
    return 0;
  }

  if (args.hasArg(OPT_version)) {
    commandLine.printVersion(llvm::outs());
    return 0;
  }

  auto triple = llvm::Triple::normalize(
      args.getLastArgValue(OPT_target_EQ, LLVM_DEFAULT_TARGET_TRIPLE));
  auto toolchain = createToolchainForTriple(commandLine, llvm::Triple(triple));
  if (!toolchain) {
    auto* arg = args.getLastArg(OPT_target_EQ);
    if (!arg) {
      commandLine.createError(pylir::Diag::UNSUPPORTED_TARGET_N, triple);
      return -1;
    }
    commandLine.createError(arg, pylir::Diag::UNSUPPORTED_TARGET_N, triple)
        .addHighlight(arg);
    return -1;
  }

  if (!commandLine)
    return -1;

  pylir::CompilerInvocation::Action action = pylir::CompilerInvocation::Link;
  if (args.hasArg(OPT_fsyntax_only)) {
    action = pylir::CompilerInvocation::SyntaxOnly;
    auto* syntaxOnly = args.getLastArg(OPT_fsyntax_only);

    auto diagActionWithIR = [&](llvm::opt::Arg* actionArg,
                                std::string_view name) {
      commandLine
          .createWarning(
              actionArg,
              pylir::Diag::N_WONT_BE_EMITTED_WHEN_ONLY_CHECKING_SYNTAX, name)
          .addHighlight(actionArg)
          .addHighlight(syntaxOnly, Diag::flags::secondaryColour);
    };

    if (auto* lastIR = args.getLastArg(OPT_emit_llvm, OPT_emit_pylir)) {
      std::string_view name;
      switch (lastIR->getOption().getID()) {
      case OPT_emit_llvm: name = "LLVM IR"; break;
      case OPT_emit_pylir: name = "Pylir IR"; break;
      }
      diagActionWithIR(lastIR, name);
    } else if (auto* lastActionModifier = args.getLastArg(OPT_S, OPT_c)) {
      diagActionWithIR(lastActionModifier,
                       lastActionModifier->getOption().getID() == OPT_S
                           ? "Assembly"
                           : "Object file");
    }
  } else if (auto* arg = args.getLastArg(OPT_S, OPT_c)) {
    action = arg->getOption().getID() == OPT_S
                 ? pylir::CompilerInvocation::Assembly
                 : pylir::CompilerInvocation::ObjectFile;
  } else if (args.hasArg(OPT_emit_llvm, OPT_emit_pylir)) {
    action = pylir::CompilerInvocation::Assembly;
  }

  if (auto* opt = args.getLastArg(OPT_O);
      opt && opt->getValue() == std::string_view{"4"} &&
      !args.hasArg(OPT_emit_pylir, OPT_emit_llvm) &&
      (action == pylir::CompilerInvocation::Assembly ||
       action == pylir::CompilerInvocation::ObjectFile) &&
      !args.hasArg(OPT_flto, OPT_fno_lto)) {
    commandLine
        .createWarning(
            opt, pylir::Diag::O4_MAY_ENABLE_LTO_COMPILER_MIGHT_OUTPUT_LLVM_IR,
            action == pylir::CompilerInvocation::Assembly ? "Assembly file"
                                                          : "Object file")
        .addHighlight(opt);
  }

  if (auto* opt = args.getLastArg(OPT_flto, OPT_fno_lto);
      opt && opt->getOption().matches(OPT_flto) &&
      !args.hasArg(OPT_emit_pylir, OPT_emit_llvm) &&
      (action == pylir::CompilerInvocation::Assembly ||
       action == pylir::CompilerInvocation::ObjectFile)) {
    commandLine
        .createWarning(
            opt, pylir::Diag::LTO_ENABLED_COMPILER_WILL_OUTPUT_LLVM_IR,
            action == pylir::CompilerInvocation::Assembly ? "Assembly file"
                                                          : "Object file")
        .addHighlight(opt);
  }

  if (args.hasMultipleArgs(OPT_INPUT)) {
    auto* second = *std::next(args.filtered(OPT_INPUT).begin());
    commandLine.createError(second, pylir::Diag::EXPECTED_ONLY_ONE_INPUT_FILE)
        .addHighlight(second);
    return -1;
  }

  auto* inputFile = args.getLastArg(OPT_INPUT);
  if (!inputFile) {
    commandLine.createError(pylir::Diag::NO_INPUT_FILE);
    return -1;
  }

  std::optional<pylir::DiagnosticsVerifier> verifier;
  if (args.hasArg(OPT_verify))
    verifier.emplace(diagnosticManager);

  pylir::CompilerInvocation invocation(verifier ? &*verifier : nullptr);
  auto result = invocation.executeAction(inputFile, commandLine, *toolchain,
                                         action, diagnosticManager);
  if (verifier)
    result = verifier->verify();

  return mlir::succeeded(result) ? 0 : -1;
}
