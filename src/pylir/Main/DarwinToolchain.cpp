// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "DarwinToolchain.hpp"

#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Program.h>

void pylir::DarwinToolchain::deduceSDKRoot(const pylir::cli::CommandLine& commandLine)
{
    const auto& args = commandLine.getArgs();

    // We need the SDK path where the system libraries are contained.
    // First, respect the users wish if specified via sysroot.
    if (auto* arg = args.getLastArg(cli::OPT_sysroot_EQ))
    {
        m_sdkRoot = arg->getValue();
        return;
    }

    // Otherwise we might be running in a xcode developer environment that defines SDKROOT.
    if (char* env = std::getenv("SDKROOT"))
    {
        if (llvm::sys::path::is_absolute(env) && llvm::sys::fs::exists(env) && llvm::StringRef(env) != "/")
        {
            m_sdkRoot = env;
            return;
        }
    }

    // As a last resort we attempt to run 'xcrun --show-sdk-path' to get the path.
    llvm::SmallString<64> outputFile;
    llvm::sys::fs::createTemporaryFile("print-sdk-path", "", outputFile);

    llvm::Optional<llvm::StringRef> redirects[] = {{""}, outputFile.str(), {""}};

    llvm::Optional<llvm::StringRef> sdkName;
    switch (m_triple.getOS())
    {
        case llvm::Triple::Darwin:
        case llvm::Triple::MacOSX: sdkName = "macosx"; break;
        default: break;
    }
    if (!sdkName)
    {
        return;
    }

    auto result = llvm::sys::ExecuteAndWait("/usr/bin/xcrun", {"/usr/bin/xcrun", "--sdk", *sdkName, "--show-sdk-path"},
                                            llvm::None, redirects, 0, 0);
    if (result != 0)
    {
        return;
    }

    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> outputBuffer = llvm::MemoryBuffer::getFile(outputFile, true);
    if (!outputBuffer)
    {
        return;
    }
    m_sdkRoot = (*outputBuffer)->getBuffer().trim();
}

pylir::DarwinToolchain::DarwinToolchain(llvm::Triple triple, const pylir::cli::CommandLine& commandLine)
    : Toolchain(std::move(triple), commandLine)
{
    deduceSDKRoot(commandLine);
}

namespace
{
llvm::StringRef getMachOArchName(const llvm::Triple& triple)
{
    switch (triple.getArch())
    {
        case llvm::Triple::aarch64_32: return "arm64_32";
        case llvm::Triple::aarch64:
            if (triple.isArm64e())
            {
                return "arm64e";
            }
            return "arm64";
        default: return triple.getArchName();
    }
}
} // namespace

bool pylir::DarwinToolchain::link(pylir::cli::CommandLine& commandLine, llvm::StringRef objectFile) const
{
    const auto& args = commandLine.getArgs();

    auto linkerInvocation = LinkerInvocationBuilder(LinkerStyle::Mac);

    llvm::VersionTuple targetVersion;
    m_triple.getMacOSXVersion(targetVersion);
    llvm::VersionTuple minTargetVersion = m_triple.getMinimumSupportedOSVersion();
    if (!minTargetVersion.empty() && minTargetVersion > targetVersion)
    {
        targetVersion = minTargetVersion;
    }

    linkerInvocation.addArg("-no_deduplicate")
        .addArg("-dynamic")
        .addArg("-arch")
        .addArg(getMachOArchName(m_triple))
        .addArg("-platform_version")
        .addArg("macos")
        .addArg(targetVersion.getAsString())
        .addArg("13.0")
        // TODO: deploy version
        .addArg("-pie", isPIE(commandLine))
        .addLLVMOptions(getLLVMOptions(args))
        .addArg("-syslibroot")
        .addArg(m_sdkRoot);

    if (auto* output = args.getLastArg(cli::OPT_o))
    {
        linkerInvocation.addOutputFile(output->getValue());
    }
    else if (auto* input = args.getLastArg(cli::OPT_INPUT))
    {
        linkerInvocation.addOutputFile(llvm::sys::path::stem(input->getValue()));
    }

    linkerInvocation
        .addLibrarySearchDirs(args.getAllArgValues(cli::OPT_L))
        .addLibrarySearchDirs(m_builtinLibrarySearchDirs)
        .addLibrarySearchDir("/","usr", "local", "lib");

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

    linkerInvocation
        .addArg(objectFile)
        .addLibrary("PylirRuntime")
        .addLibrary("PylirMarkAndSweep")
        .addLibrary("PylirRuntimeMain")
        .addLibrary("c++")
        .addLibrary("System")
        //TODO:
        .addArg("/Library/Developer/CommandLineTools/usr/lib/clang/14.0.0/lib/darwin/libclang_rt.osx.a");

    return callLinker(commandLine, std::move(linkerInvocation));
}

bool pylir::DarwinToolchain::defaultsToPIC() const
{
    return true;
}
