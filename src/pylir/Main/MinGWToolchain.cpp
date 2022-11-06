
//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "MinGWToolchain.hpp"

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Program.h>

#include "Version.hpp"

std::string pylir::MinGWToolchain::getRuntimeLibname(llvm::StringRef name) const
{
    if (m_perTargetRuntimeDir)
    {
        return "clang_rt." + name.str();
    }
    return ("clang_rt." + name.str() + "-" + m_triple.getArchName()).str();
}

void pylir::MinGWToolchain::searchForClangInstallation(const pylir::cli::CommandLine& commandLine)
{
    std::vector<std::string> candidates;
    // Always respect the users wish of the sysroot if present.
    if (auto* arg = commandLine.getArgs().getLastArg(pylir::cli::OPT_sysroot_EQ))
    {
        llvm::SmallString<32> temp;
        llvm::sys::path::append(temp, arg->getValue());
        candidates.emplace_back(temp);
    }
    else
    {
        llvm::SmallString<32> executablePath = commandLine.getExecutablePath();
        llvm::sys::path::remove_filename(executablePath);
        llvm::sys::path::append(executablePath, "..", m_triple.str());
        candidates.emplace_back(executablePath);

#ifdef _WIN32
        auto addClangFromPath = [&](llvm::Twine name)
        {
            if (auto result = llvm::sys::findProgramByName(name.str()))
            {
                // Strip filename and get out of bin.
                candidates.emplace_back(llvm::sys::path::parent_path(llvm::sys::path::parent_path(*result)));
            }
        };

        // There is no concept of a sysroot on Windows. Aka, there is no fixed directory where we'll find an
        // installation of clang. Instead, we search on the PATH.
        addClangFromPath(m_triple.getArchName() + "-w64-mingw32-clang++.exe");
        addClangFromPath(m_triple.str() + "-clang++.exe");
        addClangFromPath("clang++.exe");
#endif
    }

    pylir::Version currentVersion;
    for (auto& iter : candidates)
    {
        llvm::SmallString<32> path{iter};
        llvm::sys::path::append(path, "lib", "clang");
        if (!llvm::sys::fs::exists(path))
        {
            continue;
        }
        std::error_code ec;
        for (llvm::sys::fs::directory_iterator begin(path, ec), end; !ec && begin != end; begin = begin.increment(ec))
        {
            auto newVersion = pylir::Version::parse(llvm::sys::path::filename(begin->path()));
            if (!newVersion)
            {
                continue;
            }
            if (currentVersion > *newVersion)
            {
                continue;
            }
            currentVersion = std::move(*newVersion);
            m_sysroot = iter;
            llvm::SmallString<32> temp{begin->path()};
            llvm::sys::path::append(temp, "lib", m_triple.str());
            m_perTargetRuntimeDir = llvm::sys::fs::exists(temp);
            if (!m_perTargetRuntimeDir)
            {
                llvm::sys::path::remove_filename(temp);
                llvm::sys::path::append(temp, "windows");
            }
            m_runtimeDir = temp.str();
        }
    }
}

pylir::MinGWToolchain::MinGWToolchain(llvm::Triple triple, const cli::CommandLine& commandLine)
    : Toolchain(std::move(triple), commandLine)
{
    searchForClangInstallation(commandLine);
    if (m_sysroot.empty())
    {
        return;
    }

    m_builtinLibrarySearchDirs.emplace_back(m_runtimeDir);
    addIfExists(m_sysroot, "lib");
    addIfExists(m_sysroot, m_triple.getArchName() + "-w64-mingw32");
    addIfExists(m_sysroot, m_triple.str());
}

bool pylir::MinGWToolchain::link(cli::CommandLine& commandLine, llvm::StringRef objectFile) const
{
    const auto& args = commandLine.getArgs();

    auto linkerInvocation = LinkerInvocationBuilder(LinkerStyle::MinGW)
                                .addArg("--sysroot=" + m_sysroot, !m_sysroot.empty())
                                .addEmulation(m_triple)
                                .addLLVMOptions(getLLVMOptions(args))
                                .addArg("-Bstatic");

    if (auto* output = args.getLastArg(cli::OPT_o))
    {
        linkerInvocation.addOutputFile(output->getValue());
    }
    else if (auto* input = args.getLastArg(cli::OPT_INPUT))
    {
        llvm::SmallString<20> path(llvm::sys::path::stem(input->getValue()));
        llvm::sys::path::replace_extension(path, ".exe");
        linkerInvocation.addOutputFile(path);
    }

    linkerInvocation.addArg(findOnBuiltinPaths("crt2.o"))
        .addArg(findOnBuiltinPaths("crtbegin.o"))
        .addLibrarySearchDirs(args.getAllArgValues(cli::OPT_L))
        .addLibrarySearchDirs(m_builtinLibrarySearchDirs)
        .addArg(objectFile);

    for (auto* arg : args)
    {
        if (arg->getOption().matches(cli::OPT_l))
        {
            linkerInvocation.addLibrary(arg->getValue());
            continue;
        }
        if (arg->getOption().matches(cli::OPT_Wl))
        {
            linkerInvocation.addArgs(arg->getValues());
            continue;
        }
    }

    linkerInvocation.addArg("--start-group")
        .addLibrary("PylirRuntime")
        .addLibrary("PylirMarkAndSweep")
        .addLibrary("PylirRuntimeMain")
        .addArg("--end-group")
        .addLibrary("c++")
        .addArg("--start-group")
        .addLibrary("mingw32")
        .addLibrary(getRuntimeLibname("builtins"))
        .addLibrary("moldname")
        .addLibrary("mingwex");

    auto argValues = args.getAllArgValues(cli::OPT_l);
    if (std::none_of(argValues.begin(), argValues.end(),
                     [](llvm::StringRef ref) { return ref.startswith("msvcr") || ref.startswith("ucrt"); }))
    {
        linkerInvocation.addLibrary("msvcrt");
    }
    linkerInvocation.addLibrary("advapi32")
        .addLibrary("shell32")
        .addLibrary("user32")
        .addLibrary("kernel32")
        .addArg("--end-group")
        .addArg(findOnBuiltinPaths("crtend.o"));

    return callLinker(commandLine, std::move(linkerInvocation));
}
