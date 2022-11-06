//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/ADT/Triple.h>
#include <llvm/Support/FileSystem.h>

#include "CommandLine.hpp"
#include "LinkerInvocation.hpp"

namespace pylir
{
class Toolchain
{
protected:
    llvm::Triple m_triple;
    std::vector<std::string> m_programPaths;
    std::vector<std::string> m_builtinLibrarySearchDirs;

    ///
    std::string findOnBuiltinPaths(llvm::StringRef file) const;

    ///
    template <class... Args>
    void addIfExists(Args&&... args)
    {
        llvm::SmallString<100> temp;
        llvm::sys::path::append(temp, std::forward<Args>(args)...);
        if (llvm::sys::fs::exists(temp))
        {
            m_builtinLibrarySearchDirs.emplace_back(temp);
        }
    }

    [[nodiscard]] bool callLinker(cli::CommandLine& commandLine,
                                  LinkerInvocationBuilder&& linkerInvocationBuilder) const;

    [[nodiscard]] virtual bool defaultsToPIE() const
    {
        return false;
    }

public:
    explicit Toolchain(llvm::Triple triple, const cli::CommandLine& commandLine);

    virtual ~Toolchain() = default;

    Toolchain(const Toolchain&) = delete;
    Toolchain& operator=(const Toolchain&) = delete;
    Toolchain(Toolchain&&) = delete;
    Toolchain& operator=(Toolchain&&) = delete;

    [[nodiscard]] std::vector<std::string> getLLVMOptions(const llvm::opt::InputArgList& args) const;

    [[nodiscard]] virtual bool link(cli::CommandLine& commandLine, llvm::StringRef objectFile) const = 0;

    [[nodiscard]] bool isPIE(const pylir::cli::CommandLine& commandLine) const;

    [[nodiscard]] virtual bool defaultsToPIC() const
    {
        return false;
    }
};
} // namespace pylir
