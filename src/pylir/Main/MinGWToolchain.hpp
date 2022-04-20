// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "Toolchain.hpp"

namespace pylir
{
class MinGWToolchain : public Toolchain
{
    std::string m_sysroot;
    std::string m_subdir;

protected:
    [[nodiscard]] Stdlib defaultStdlib() const override;

    [[nodiscard]] RTLib defaultRTLib() const override;

    [[nodiscard]] bool defaultsToPIC() const override;

public:
    explicit MinGWToolchain(const llvm::Triple& triple, const cli::CommandLine& commandLine);

    [[nodiscard]] bool link(const cli::CommandLine& commandLine, llvm::StringRef objectFile) const override;
};
} // namespace pylir
