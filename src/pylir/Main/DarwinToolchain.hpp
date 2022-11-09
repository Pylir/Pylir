// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/Support/VersionTuple.h>

#include "Toolchain.hpp"

namespace pylir
{
class DarwinToolchain : public Toolchain
{
    std::string m_sdkRoot;
    llvm::VersionTuple m_sdkVersion;
    llvm::VersionTuple m_maxDeployVersion;

    void deduceSDKRoot(const cli::CommandLine& commandLine);

public:

    [[nodiscard]] bool defaultsToPIC() const override;

    DarwinToolchain(llvm::Triple triple, const cli::CommandLine& commandLine);

    bool link(cli::CommandLine& commandLine, llvm::StringRef objectFile) const override;
};
} // namespace pylir