
//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "Toolchain.hpp"

namespace pylir {
class LinuxToolchain : public Toolchain {
  std::string m_sysroot;
  ClangInstallation m_clangInstallation;

  void findClangInstallation(const cli::CommandLine& commandLine);

public:
  explicit LinuxToolchain(llvm::Triple triple, cli::CommandLine& commandLine);

  [[nodiscard]] bool link(cli::CommandLine& commandLine,
                          llvm::StringRef objectFile) const override;
};
} // namespace pylir
