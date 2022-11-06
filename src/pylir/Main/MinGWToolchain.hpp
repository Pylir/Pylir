//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "Toolchain.hpp"

namespace pylir
{
class MinGWToolchain : public Toolchain
{
    std::string m_sysroot;
    bool m_perTargetRuntimeDir{};
    std::string m_runtimeDir;

    [[nodiscard]] std::string getRuntimeLibname(llvm::StringRef name) const;

    void searchForClangInstallation(const cli::CommandLine& commandLine);

protected:

    [[nodiscard]] bool defaultsToPIC() const override
    {
        return true;
    }

public:
    MinGWToolchain(llvm::Triple triple, const cli::CommandLine& commandLine);

    [[nodiscard]] bool link(cli::CommandLine& commandLine, llvm::StringRef objectFile) const override;
};
} // namespace pylir
