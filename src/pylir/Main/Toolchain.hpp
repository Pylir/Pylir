// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/ADT/Triple.h>

#include "CommandLine.hpp"

namespace pylir
{
class Toolchain
{
protected:
    llvm::Triple m_triple;
    std::vector<std::string> m_programPaths;

    enum class LinkerStyle
    {
        MSVC,
        GNU,
        Mac,
        Wasm,
    };

    [[nodiscard]] bool callLinker(const pylir::cli::CommandLine& commandLine, LinkerStyle style,
                                  llvm::ArrayRef<std::string> arguments) const;

    enum class Stdlib
    {
        libstdcpp,
        libcpp,
    };

    [[nodiscard]] virtual Stdlib defaultStdlib() const = 0;

    [[nodiscard]] Stdlib getStdlib(const pylir::cli::CommandLine& commandLine) const;

    enum class RTLib
    {
        compiler_rt,
        libgcc,
    };

    [[nodiscard]] virtual RTLib defaultRTLib() const = 0;

    [[nodiscard]] RTLib getRTLib(const pylir::cli::CommandLine& commandLine) const;

    [[nodiscard]] virtual bool defaultsToPIE() const
    {
        return false;
    }

public:
    explicit Toolchain(llvm::Triple triple) : m_triple(std::move(triple)) {}

    virtual ~Toolchain() = default;

    Toolchain(const Toolchain&) = delete;
    Toolchain& operator=(const Toolchain&) = delete;
    Toolchain(Toolchain&&) = delete;
    Toolchain& operator=(Toolchain&&) = delete;

    [[nodiscard]] std::vector<std::string> getLLVMOptions(const llvm::opt::InputArgList& args) const;

    [[nodiscard]] virtual bool link(const pylir::cli::CommandLine& commandLine, llvm::StringRef objectFile) const = 0;

    [[nodiscard]] bool isPIE(const pylir::cli::CommandLine& commandLine) const;

    [[nodiscard]] virtual bool defaultsToPIC() const
    {
        return false;
    }
};
} // namespace pylir
