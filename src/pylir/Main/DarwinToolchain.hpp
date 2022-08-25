// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "Toolchain.hpp"

namespace pylir
{
class DarwinToolchain : public Toolchain
{
protected:
    [[nodiscard]] Stdlib defaultStdlib() const override
    {
        return Stdlib::libcpp;
    }

    [[nodiscard]] RTLib defaultRTLib() const override
    {
        return RTLib::compiler_rt;
    }

public:
    explicit DarwinToolchain(const llvm::Triple& triple, const cli::CommandLine& commandLine);

    [[nodiscard]] bool link(cli::CommandLine& commandLine, llvm::StringRef objectFile) const override;
};
} // namespace pylir
