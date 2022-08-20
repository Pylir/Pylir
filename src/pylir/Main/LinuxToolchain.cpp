// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "LinuxToolchain.hpp"

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>

#include "Distro.hpp"
#include "Version.hpp"

namespace
{
struct Multilib
{
    std::string gccSuffix;
    std::string osSuffix;
    std::string includeSuffix;
};

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

const char* getEmulation(const llvm::Triple& triple, const pylir::cli::CommandLine&)
{
    switch (triple.getArch())
    {
        case llvm::Triple::x86_64: return "elf_x86_64";
        default: return nullptr;
    }
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

pylir::LinuxToolchain::LinuxToolchain(const llvm::Triple& triple, const cli::CommandLine&) : Toolchain(triple) {}

bool pylir::LinuxToolchain::link(const pylir::cli::CommandLine& commandLine, llvm::StringRef objectFile) const
{
    const auto& args = commandLine.getArgs();
    auto gccInstall = findGCCInstallation(m_triple, commandLine);
    if (!gccInstall)
    {
        llvm::errs() << pylir::Diag::formatLine(Diag::Severity::Error, "Failed to find a GCC installation");
        return false;
    }
    std::vector<std::string> arguments;
    auto sysroot = commandLine.getArgs().getLastArgValue(pylir::cli::OPT_sysroot_EQ, PYLIR_DEFAULT_SYSROOT);
    if (!sysroot.empty())
    {
        arguments.push_back(("--sysroot=" + sysroot).str());
    }
    if (isPIE(commandLine))
    {
        arguments.emplace_back("-pie");
    }
    Distro distro(m_triple);
    if (distro.IsAlpineLinux())
    {
        arguments.emplace_back("-z");
        arguments.emplace_back("now");
    }
    if (distro.IsOpenSUSE() || distro.IsUbuntu() || distro.IsAlpineLinux())
    {
        arguments.emplace_back("-z");
        arguments.emplace_back("relro");
    }

    if (distro.IsRedhat() || distro.IsOpenSUSE() || distro.IsAlpineLinux()
        || (distro.IsUbuntu() && distro >= Distro::UbuntuMaverick))
    {
        arguments.emplace_back("--hash-style=gnu");
    }

    if (distro.IsDebian() || distro.IsOpenSUSE() || distro == Distro::UbuntuLucid || distro == Distro::UbuntuJaunty
        || distro == Distro::UbuntuKarmic)
    {
        arguments.emplace_back("--hash-style=both");
    }

    if (distro.IsOpenSUSE())
    {
        arguments.emplace_back("--enable-new-dtags");
    }

    arguments.emplace_back("--eh-frame-hdr");
    const auto* emulation = getEmulation(m_triple, commandLine);
    if (!emulation)
    {
        llvm::errs() << pylir::Diag::formatLine(Diag::Severity::Error,
                                                fmt::format("Missing emulation for target '{}'", m_triple.str()));
        return false;
    }
    arguments.emplace_back("-m");
    arguments.emplace_back(emulation);

    arguments.emplace_back("-dynamic-linker");
    arguments.emplace_back(getDynamicLinker(m_triple, commandLine));

    for (auto& iter : getLLVMOptions(args))
    {
        arguments.push_back("--mllvm=" + iter);
    }

    if (auto* output = args.getLastArg(pylir::cli::OPT_o))
    {
        arguments.emplace_back("-o");
        arguments.emplace_back(output->getValue());
    }
    else if (auto* input = args.getLastArg(pylir::cli::OPT_INPUT))
    {
        llvm::SmallString<20> path(input->getValue());
        llvm::sys::path::replace_extension(path, "");
        arguments.emplace_back("-o");
        arguments.emplace_back(path);
    }
    auto sep = llvm::sys::path::get_separator();
    std::vector<llvm::SmallString<32>> builtinPaths;
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

    arguments.push_back(findOnBuiltinPath("crt1.o"));
    arguments.push_back(findOnBuiltinPath("crti.o"));
    arguments.push_back(findOnBuiltinPath("crtbegin.o"));

    for (auto& iter : args.getAllArgValues(pylir::cli::OPT_L))
    {
        arguments.push_back("-L" + iter);
    }

    for (auto& iter : builtinPaths)
    {
        arguments.push_back(("-L" + iter).str());
    }

    for (auto* arg : args)
    {
        if (arg->getOption().matches(pylir::cli::OPT_l))
        {
            arguments.push_back("-l" + std::string(arg->getValue()));
            continue;
        }
        if (arg->getOption().matches(pylir::cli::OPT_Wl))
        {
            std::copy(arg->getValues().begin(), arg->getValues().end(), std::back_inserter(arguments));
            continue;
        }
    }

    arguments.push_back(objectFile.str());
    llvm::SmallString<10> executablePath = commandLine.getExecutablePath();
    llvm::sys::path::remove_filename(executablePath);
    llvm::sys::path::append(executablePath, "..", "lib", "pylir", m_triple.str());
    llvm::sys::path::append(executablePath, "libPylirRuntime.a");
    arguments.emplace_back("--start-group");
    arguments.emplace_back(executablePath);
    llvm::sys::path::remove_filename(executablePath);
    // TODO: Change to respect the command line option
    llvm::sys::path::append(executablePath, "libPylirMarkAndSweep.a");
    arguments.emplace_back(executablePath);
    llvm::sys::path::remove_filename(executablePath);
    llvm::sys::path::append(executablePath, "libPylirRuntimeMain.a");
    arguments.emplace_back("--whole-archive");
    arguments.emplace_back(executablePath);
    arguments.emplace_back("--no-whole-archive");
    arguments.emplace_back("--end-group");
    arguments.emplace_back("-lunwind");

    switch (getStdlib(commandLine))
    {
        case Stdlib::libstdcpp: arguments.emplace_back("-lstdc++"); break;
        case Stdlib::libcpp: arguments.emplace_back("-lc++"); break;
    }

    arguments.emplace_back("-lm");

    arguments.emplace_back("-lgcc_s");
    arguments.emplace_back("-lgcc");
    arguments.emplace_back("-lc");
    //    arguments.emplace_back("-lgcc_s");
    //    arguments.emplace_back("-lgcc");

    arguments.push_back(findOnBuiltinPath("crtend.o"));
    arguments.push_back(findOnBuiltinPath("crtn.o"));

    return callLinker(commandLine, Toolchain::LinkerStyle::GNU, arguments);
}
