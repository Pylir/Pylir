
//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "MinGWToolchain.hpp"

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Program.h>

#include "Version.hpp"

namespace
{
std::optional<std::string> relativeSubdir(const llvm::Triple& triple, const pylir::cli::CommandLine& commandLine,
                                          std::string& subdir)
{
    llvm::SmallVector<llvm::SmallString<32>, 2> subdirs;
    subdirs.emplace_back(triple.str());
    subdirs.emplace_back(triple.getArchName());
    subdirs[1] += "-w64-mingw32";
    llvm::SmallString<10> executablePath = commandLine.getExecutablePath();
    executablePath = llvm::sys::path::parent_path(executablePath).str();
    executablePath = llvm::sys::path::parent_path(executablePath).str();
    for (auto& candidateSubdir : subdirs)
    {
        if (llvm::sys::fs::is_directory(executablePath + llvm::sys::path::get_separator() + candidateSubdir))
        {
            subdir = candidateSubdir.str();
            return (executablePath + llvm::sys::path::get_separator() + candidateSubdir).str();
        }
    }
    return {};
}
} // namespace

pylir::MinGWToolchain::MinGWToolchain(llvm::Triple triple, const cli::CommandLine& commandLine)
    : Toolchain(std::move(triple), commandLine)
{
    m_sysroot = commandLine.getArgs().getLastArgValue(cli::OPT_sysroot_EQ, PYLIR_DEFAULT_SYSROOT);
    if (m_sysroot.empty() || !llvm::sys::fs::is_directory(m_sysroot))
    {
        if (auto opt = relativeSubdir(m_triple, commandLine, m_subdir))
        {
            m_sysroot = std::move(*opt);
        }
        else
        {
            std::string gccName = (m_triple.getArchName() + "-w64-mingw32-gcc").str();
            if (auto result = llvm::sys::findProgramByName(gccName))
            {
                m_subdir = (m_triple.getArchName() + "-w64-ming32").str();
                m_sysroot = llvm::sys::path::parent_path(gccName);
            }
        }
    }
    else
    {
        llvm::SmallVector<llvm::SmallString<32>, 2> subdirs;
        subdirs.emplace_back(m_triple.str());
        subdirs.emplace_back(m_triple.getArchName());
        subdirs[1] += "-w64-mingw32";
        for (auto& candidateSubdir : subdirs)
        {
            if (llvm::sys::fs::is_directory(m_sysroot + llvm::sys::path::get_separator() + candidateSubdir))
            {
                m_subdir = candidateSubdir.str();
                break;
            }
        }
    }
}

namespace
{

std::optional<std::string> findGCCLib(const llvm::Triple& triple, llvm::StringRef sysroot)
{
    llvm::SmallVector<llvm::SmallString<32>> gccLibCandidates;
    gccLibCandidates.emplace_back(triple.getArchName());
    gccLibCandidates[0] += "-w64-mingw32";
    gccLibCandidates.emplace_back("ming32");
    for (const auto& libCandidate : {"lib", "lib64"})
    {
        for (auto& gccLib : gccLibCandidates)
        {
            llvm::SmallString<1024> libdir{sysroot};
            llvm::sys::path::append(libdir, libCandidate, "gcc", gccLib);
            pylir::Version version;
            std::error_code ec;
            for (llvm::sys::fs::directory_iterator iter(libdir, ec), end; !ec && iter != end; iter = iter.increment(ec))
            {
                auto newVersion = pylir::Version::parse(llvm::sys::path::filename(iter->path()));
                if (!newVersion)
                {
                    continue;
                }
                version = std::max(version, *newVersion);
            }
            if (version.majorVersion == -1)
            {
                continue;
            }
            llvm::sys::path::append(libdir, version.original);
            return std::optional<std::string>{std::in_place, libdir};
        }
    }
    return {};
}

std::optional<std::string> findClangResourceDir(const llvm::Triple& triple, llvm::StringRef sysroot)
{
    llvm::SmallString<1024> path{sysroot};
    llvm::sys::path::append(path, "lib", "clang");
    pylir::Version version;
    std::error_code ec;
    for (llvm::sys::fs::directory_iterator iter(path, ec), end; !ec && iter != end; iter = iter.increment(ec))
    {
        auto newVersion = pylir::Version::parse(llvm::sys::path::filename(iter->path()));
        if (!newVersion)
        {
            continue;
        }
        version = std::max(version, *newVersion);
    }
    if (version.majorVersion == -1)
    {
        return {};
    }
    llvm::sys::path::append(path, version.original, "lib");
    auto sep = llvm::sys::path::get_separator();
    if (llvm::sys::fs::exists(path + sep + triple.str()))
    {
        return (path + sep + triple.str()).str();
    }
    if (llvm::sys::fs::exists(path + sep + triple.getOSName()))
    {
        return (path + sep + triple.getOSName()).str();
    }
    return {};
}

} // namespace

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

    auto gccLib = findGCCLib(m_triple, m_sysroot);
    auto sep = llvm::sys::path::get_separator();
    llvm::SmallString<20> sysRootAndSubDir(m_sysroot);
    if (!m_subdir.empty())
    {
        sysRootAndSubDir += sep;
        sysRootAndSubDir += m_subdir;
    }
    sysRootAndSubDir += sep;
    sysRootAndSubDir += "lib";

    llvm::SmallString<20> path(gccLib.value_or(static_cast<std::string>(sysRootAndSubDir)));
    linkerInvocation.addArg(path + sep + "crt2.o")
        .addArg(path + sep + "crtbegin.o")
        .addLibrarySearchDirs(args.getAllArgValues(cli::OPT_L));

    if (gccLib)
    {
        linkerInvocation.addLibrarySearchDir(*gccLib);
    }
    linkerInvocation.addLibrarySearchDir(sysRootAndSubDir)
        .addLibrarySearchDir(m_sysroot, "lib")
        .addLibrarySearchDir(m_sysroot, "lib", m_triple.str())
        .addLibrarySearchDir(m_sysroot, "sys-root", "mingw", "lib");

    auto clangRTPath = findClangResourceDir(m_triple, m_sysroot);
    if (clangRTPath)
    {
        linkerInvocation.addLibrarySearchDir(*clangRTPath);
    }

    linkerInvocation.addArg(objectFile);

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
        .addLibrary("mingw32");

    if (!clangRTPath || llvm::sys::fs::exists(*clangRTPath + sep + "libclang_rt.builtins.a"))
    {
        linkerInvocation.addLibrary("clang-rt.builtins");
    }
    else
    {
        linkerInvocation.addLibrary("clang-rt.builtins-" + m_triple.getArchName());
    }

    linkerInvocation.addLibrary("moldname").addLibrary("mingwex");

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
        .addArg(path + sep + "crtend.o");

    return callLinker(commandLine, std::move(linkerInvocation));
}
