
//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "Toolchain.hpp"

namespace pylir
{
class LinuxToolchain : public Toolchain
{
protected:
    [[nodiscard]] Stdlib defaultStdlib() const override
    {
        return Stdlib::libstdcpp;
    }

    [[nodiscard]] RTLib defaultRTLib() const override
    {
        return RTLib::libgcc;
    }

public:
    explicit LinuxToolchain(const llvm::Triple& triple, const cli::CommandLine& commandLine);

    [[nodiscard]] bool link(cli::CommandLine& commandLine, llvm::StringRef objectFile) const override;
};
} // namespace pylir
