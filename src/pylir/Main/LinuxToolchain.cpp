//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "LinuxToolchain.hpp"

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>

#include "Distro.hpp"
#include "Version.hpp"

namespace
{

struct GCCInstallation
{
    std::string libPath;
    std::string gccLibPath;
    llvm::Triple gccTriple;
};

void collectLibDirsAndTriples(const llvm::Triple& triple, llvm::SmallVectorImpl<llvm::StringRef>& libDirs,
                              llvm::SmallVectorImpl<llvm::StringRef>& triples)
{
    libDirs.emplace_back("lib");
    if (triple.isArch64Bit())
    {
        libDirs.emplace_back("lib64");
    }
    else if (triple.isArch32Bit())
    {
        libDirs.emplace_back("lib32");
    }
    triples.emplace_back(triple.str());
    switch (triple.getArch())
    {
        case llvm::Triple::x86_64:
        {
            triples.append({"x86_64-linux-gnu", "x86_64-unknown-linux-gnu", "x86_64-pc-linux-gnu",
                            "x86_64-redhat-linux6E", "x86_64-redhat-linux", "x86_64-suse-linux",
                            "x86_64-manbo-linux-gnu", "x86_64-linux-gnu", "x86_64-slackware-linux",
                            "x86_64-unknown-linux", "x86_64-amazon-linux"});
            break;
        }
        default: break;
    }
}

std::optional<GCCInstallation> findGCCInstallation(const llvm::Triple& triple,
                                                   const pylir::cli::CommandLine& commandLine)
{
    llvm::SmallVector<llvm::StringRef, 4> candidateLibDirs;
    llvm::SmallVector<llvm::StringRef, 16> candidateTriples;
    collectLibDirsAndTriples(triple, candidateLibDirs, candidateTriples);

    llvm::SmallVector<std::string, 8> prefixes;
    auto sysroot = commandLine.getArgs().getLastArgValue(pylir::cli::OPT_sysroot_EQ, PYLIR_DEFAULT_SYSROOT);
    if (!sysroot.empty())
    {
        prefixes.push_back(sysroot.str());
        prefixes.push_back((sysroot + llvm::sys::path::get_separator() + "usr").str());
    }
    llvm::SmallString<32> executablePath = commandLine.getExecutablePath();
    llvm::sys::path::remove_filename(executablePath);
    prefixes.push_back(llvm::sys::path::parent_path(executablePath).str());
    if (sysroot.empty())
    {
        prefixes.push_back("/usr");
    }

    pylir::Version currentVersion;
    std::string libPath;
    std::string gccLibPath;
    llvm::Triple resultTriple;
    for (auto& prefix : prefixes)
    {
        if (!llvm::sys::fs::exists(prefix))
        {
            continue;
        }
        for (auto suffix : candidateLibDirs)
        {
            for (auto gccTriple : candidateTriples)
            {
                llvm::SmallString<32> path{prefix};
                llvm::sys::path::append(path, suffix, "gcc", gccTriple);
                if (!llvm::sys::fs::exists(path))
                {
                    continue;
                }
                std::error_code ec;
                for (llvm::sys::fs::directory_iterator iter(path, ec), end; !ec && iter != end;
                     iter = iter.increment(ec))
                {
                    auto newVersion = pylir::Version::parse(llvm::sys::path::filename(iter->path()));
                    if (!newVersion)
                    {
                        continue;
                    }
                    if (currentVersion < *newVersion)
                    {
                        currentVersion = std::move(*newVersion);
                        libPath = (prefix + llvm::sys::path::get_separator() + suffix).str();
                        gccLibPath = iter->path();
                        resultTriple = llvm::Triple(gccTriple);
                    }
                }
            }
        }
    }
    if (!currentVersion)
    {
        return {};
    }
    return GCCInstallation{std::move(libPath), std::move(gccLibPath), std::move(resultTriple)};
}

const char* getDynamicLinker(const llvm::Triple& triple, const pylir::cli::CommandLine&)
{
    switch (triple.getArch())
    {
        case llvm::Triple::x86_64: return "/lib64/ld-linux-x86-64.so.2";
        default: return nullptr;
    }
}

} // namespace

pylir::LinuxToolchain::LinuxToolchain(llvm::Triple triple, const cli::CommandLine& commandLine)
    : Toolchain(std::move(triple), commandLine)
{
}

bool pylir::LinuxToolchain::link(cli::CommandLine& commandLine, llvm::StringRef objectFile) const
{
    const auto& args = commandLine.getArgs();
    auto gccInstall = findGCCInstallation(m_triple, commandLine);
    if (!gccInstall)
    {
        commandLine.createError("Failed to find a GCC installation");
        return false;
    }

    auto sysroot = commandLine.getArgs().getLastArgValue(cli::OPT_sysroot_EQ, PYLIR_DEFAULT_SYSROOT);

    auto linkerInvocation = LinkerInvocationBuilder(LinkerStyle::ELF);
    linkerInvocation.addArg("--sysroot=" + sysroot, !sysroot.empty()).addArg("-pie", isPIE(commandLine));

    Distro distro(m_triple);
    if (distro.isAlpineLinux())
    {
        linkerInvocation.addArgs("-z", "now");
    }
    if (distro.isOpenSuse() || distro.isUbuntu() || distro.isAlpineLinux())
    {
        linkerInvocation.addArgs("-z", "relro");
    }

    if (distro.isRedhat() || distro.isOpenSuse() || distro.isAlpineLinux()
        || (distro.isUbuntu() && distro >= Distro::UbuntuMaverick))
    {
        linkerInvocation.addArg("--hash-style=gnu");
    }

    if (distro.isDebian() || distro.isOpenSuse() || distro == Distro::UbuntuLucid || distro == Distro::UbuntuJaunty
        || distro == Distro::UbuntuKarmic)
    {
        linkerInvocation.addArg("--hash-style=both");
    }

    linkerInvocation.addArg("--enable-new-dtags", distro.isOpenSuse())
        .addArg("--eh-frame-hdr")
        .addEmulation(m_triple)
        .addArgs("-dynamic-linker", getDynamicLinker(m_triple, commandLine))
        .addLLVMOptions(getLLVMOptions(args));

    if (auto* output = args.getLastArg(cli::OPT_o))
    {
        linkerInvocation.addOutputFile(output->getValue());
    }
    else if (auto* input = args.getLastArg(cli::OPT_INPUT))
    {
        linkerInvocation.addOutputFile(llvm::sys::path::stem(input->getValue()));
    }

    auto sep = llvm::sys::path::get_separator();
    std::vector<std::string> builtinPaths;
    builtinPaths.emplace_back(gccInstall->libPath);
    builtinPaths.emplace_back(gccInstall->gccLibPath);
    if (m_triple.isArch64Bit())
    {
        if (llvm::sys::fs::exists(gccInstall->libPath + sep + ".." + sep + "lib64"))
        {
            builtinPaths.emplace_back((gccInstall->libPath + sep + ".." + sep + "lib64").str());
        }
        if (llvm::sys::fs::exists(gccInstall->libPath + sep + ".." + sep + gccInstall->gccTriple.str() + sep + "lib64"))
        {
            builtinPaths.emplace_back(
                (gccInstall->libPath + sep + ".." + sep + gccInstall->gccTriple.str() + sep + "lib64").str());
        }
    }
    if (llvm::sys::fs::exists("/lib/" + gccInstall->gccTriple.str()))
    {
        builtinPaths.emplace_back("/lib/" + gccInstall->gccTriple.str());
    }
    if (llvm::sys::fs::exists("/usr/lib/" + gccInstall->gccTriple.str()))
    {
        builtinPaths.emplace_back("/usr/lib/" + gccInstall->gccTriple.str());
    }

    auto findOnBuiltinPath = [&](llvm::StringRef builtin) -> std::string
    {
        for (auto& iter : builtinPaths)
        {
            if (llvm::sys::fs::exists(iter + sep + builtin))
            {
                return (iter + sep + builtin).str();
            }
        }
        return builtin.str();
    };

    linkerInvocation.addArg(findOnBuiltinPath("crt1.o"))
        .addArg(findOnBuiltinPath("crti.o"))
        .addArg(findOnBuiltinPath("crtbegin.o"))
        .addLibrarySearchDirs(args.getAllArgValues(cli::OPT_L))
        .addLibrarySearchDirs(m_builtinLibrarySearchDirs)
        .addLibrarySearchDirs(builtinPaths);

    // Make sure the order of -l and -Wl are preserved.
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

    linkerInvocation.addArg(objectFile)
        .addArg("--start-group")
        .addLibrary("PylirRuntime")
        .addLibrary("PylirMarkAndSweep")
        .addLibrary("PylirRuntimeMain")
        .addArg("--end-group")
        .addLibrary("unwind")
        .addLibrary("stdc++")
        .addLibrary("m")
        .addLibrary("gcc_s")
        .addLibrary("gcc")
        .addLibrary("c")
        .addArg(findOnBuiltinPath("crtend.o"))
        .addArg(findOnBuiltinPath("crtn.o"));

    return callLinker(commandLine, std::move(linkerInvocation));
}
