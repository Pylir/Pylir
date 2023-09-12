// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/VersionTuple.h>

#include "Toolchain.hpp"

namespace pylir {
class DarwinToolchain : public Toolchain {
  std::string m_sdkRoot;
  std::optional<llvm::VersionTuple> m_sdkVersion;
  ClangInstallation m_clangInstallation;

  void deduceSDKRoot(const cli::CommandLine& commandLine);

  void searchForClangInstallation();

  bool readSDKSettings(llvm::MemoryBuffer& buffer);

public:
  [[nodiscard]] bool defaultsToPIC() const override;

  DarwinToolchain(llvm::Triple triple, cli::CommandLine& commandLine);

  bool link(cli::CommandLine& commandLine,
            llvm::StringRef objectFile) const override;
};
} // namespace pylir
