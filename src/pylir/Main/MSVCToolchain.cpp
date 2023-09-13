//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "MSVCToolchain.hpp"

#include <llvm/Support/Path.h>

bool pylir::MSVCToolchain::link(cli::CommandLine& commandLine,
                                llvm::StringRef objectFile) const {
  const auto& args = commandLine.getArgs();

  auto linkerInvocation = LinkerInvocationBuilder(LinkerStyle::MSVC);

  linkerInvocation.addLLVMOptions(getLLVMOptions(args))
      .addLibrarySearchDirs(args.getAllArgValues(cli::OPT_L))
      .addArg("-nologo")
      .addArg("/debug", args.getLastArgValue(cli::OPT_g, "0") != "0");

  std::string outputFile;
  if (auto* output = args.getLastArg(cli::OPT_o)) {
    linkerInvocation.addOutputFile(output->getValue());
  } else if (auto* input = args.getLastArg(cli::OPT_INPUT)) {
    llvm::SmallString<20> path(llvm::sys::path::stem(input->getValue()));
    llvm::sys::path::replace_extension(path, ".exe");
    linkerInvocation.addOutputFile(path);
  }

  // Make sure the order of -l and -Wl are preserved.
  for (auto* arg : args) {
    if (arg->getOption().matches(cli::OPT_l)) {
      linkerInvocation.addLibrary(arg->getValue());
      continue;
    }
    if (arg->getOption().matches(cli::OPT_Wl)) {
      linkerInvocation.addArgs(arg->getValues());
      continue;
    }
  }

  linkerInvocation.addArgs(args.getAllArgValues(cli::OPT_Wl))
      .addArg(objectFile);

  linkerInvocation.addLibrarySearchDirs(m_builtinLibrarySearchDirs)
      .addLibrary("PylirRuntime")
      .addLibrary("PylirMarkAndSweep")
      .addLibrary("PylirRuntimeMain");

  return callLinker(commandLine, std::move(linkerInvocation));
}
