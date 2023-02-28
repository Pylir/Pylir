//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/Support/FileSystem.h>
#include <llvm/TargetParser/Triple.h>

#include "CommandLine.hpp"
#include "LinkerInvocation.hpp"

namespace pylir
{
class Toolchain
{
    std::vector<std::string> m_programPaths;

protected:
    llvm::Triple m_triple;
    std::vector<std::string> m_builtinLibrarySearchDirs;
    bool m_wantsAddressSanitizer = false;
    bool m_wantsThreadSanitizer = false;
    bool m_wantsUndefinedSanitizer = false;

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

    /// Processes any commandline flags related to enabling sanitizers and sets internal state accordingly.
    /// Returns false if any errors were emitted.
    bool parseSanitizers(cli::CommandLine& commandLine);

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

/// Class containing information about a Clang installation installed on the users system.
class ClangInstallation
{
    /// Whether the clang installation has a per-target runtime directory layout.
    bool m_perTargetRuntimeDir{};
    /// The runtime directory of the clang installation where clangs runtimes are installed.
    std::string m_runtimeDir;
    /// The root directory of the clang installation. It contains the 'bin' and 'lib' directories.
    std::string m_rootDir;

    ClangInstallation(bool perTargetRuntimeDir, std::string runtimeDir, std::string rootDir)
        : m_perTargetRuntimeDir(perTargetRuntimeDir), m_runtimeDir(std::move(runtimeDir)), m_rootDir(std::move(rootDir))
    {
    }

public:
    ClangInstallation() = default;

    /// Searches for the newest Clang installation (that is, newest version), within all 'rootDirCandidates'. These are
    /// assumed to be the root directory of the clang installation. In other words, clang itself would be in the 'bin'
    /// of the root directory.
    /// 'triple' is used to return the runtime directory for the given target triple.
    static ClangInstallation searchForClangInstallation(llvm::ArrayRef<std::string> rootDirCandidates,
                                                        const llvm::Triple& triple);

    /// Returns true if the clang installation has a per target runtime directory.
    [[nodiscard]] bool hasPerTargetRuntimeDir() const
    {
        return m_perTargetRuntimeDir;
    }

    /// Returns the runtime directory of the clang installation.
    [[nodiscard]] llvm::StringRef getRuntimeDir() const
    {
        return m_runtimeDir;
    }

    /// Returns the root directory of the clang installation.
    [[nodiscard]] llvm::StringRef getRootDir() const
    {
        return m_rootDir;
    }

    /// Forms the library name for one of clangs runtime libraries called 'name'. The exact name may differ in suffixes
    /// and prefixes, based on clang installation, but generally has the form: clang_rt.<name>.
    [[nodiscard]] std::string getRuntimeLibname(llvm::StringRef name, const llvm::Triple& triple) const;
};

} // namespace pylir
