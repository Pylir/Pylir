//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Toolchain.hpp"

#include <llvm/Support/Program.h>

#include <pylir/Diagnostics/DiagnosticMessages.hpp>

#ifdef PYLIR_EMBEDDED_LLD
    #include <lld/Common/Driver.h>
#endif

bool pylir::Toolchain::callLinker(cli::CommandLine& commandLine, LinkerStyle style,
                                  llvm::ArrayRef<std::string> arguments) const
{
    const auto& args = commandLine.getArgs();
    std::string linkerPath;
    if (auto* arg = args.getLastArg(pylir::cli::OPT_lld_path_EQ))
    {
        linkerPath = arg->getValue();
    }
#ifndef PYLIR_EMBEDDED_LLD
    else
#else
    else if (!args.hasFlag(pylir::cli::OPT_fintegrated_lld, pylir::cli::OPT_fno_integrated_lld, true))
#endif
    {
        std::vector<llvm::StringRef> candidates;
        switch (style)
        {
            case LinkerStyle::MSVC: candidates = {"lld-link"}; break;
            case LinkerStyle::GNU: candidates = {"ld.lld"}; break;
            case LinkerStyle::Mac: candidates = {"/usr/bin/ld", "/opt/homebrew/opt/llvm/bin/ld64.lld"}; break;
            case LinkerStyle::Wasm: candidates = {"wasm-lld"}; break;
        }
        std::vector<std::string> attempts;
        for (auto iter : candidates)
        {
            std::vector<llvm::StringRef> refs(m_programPaths.begin(), m_programPaths.end());
            std::string variant = iter.str();
            attempts.push_back(variant);
            auto result = llvm::sys::findProgramByName(variant, refs);
            if (result)
            {
                linkerPath = std::move(*result);
                break;
            }
            result = llvm::sys::findProgramByName(variant);
            if (result)
            {
                linkerPath = std::move(*result);
                break;
            }
            variant = (m_triple.str() + "-" + iter).str();
            attempts.push_back(variant);
            result = llvm::sys::findProgramByName(variant, refs);
            if (result)
            {
                linkerPath = std::move(*result);
                break;
            }
            result = llvm::sys::findProgramByName(variant);
            if (result)
            {
                linkerPath = std::move(*result);
                break;
            }
        }
        if (linkerPath.empty())
        {
            commandLine.createError(pylir::Diag::FAILED_TO_FIND_LINKER)
                .addNote(pylir::Diag::ATTEMPTED_N, fmt::join(attempts, ", "));
            return false;
        }
    }
#ifdef PYLIR_EMBEDDED_LLD
    if (linkerPath.empty())
    {
        if (commandLine.verbose() || commandLine.onlyPrint())
        {
            llvm::errs() << "<builtin-";
            switch (style)
            {
                case LinkerStyle::MSVC: llvm::errs() << "lld-link"; break;
                case LinkerStyle::GNU: llvm::errs() << "ld.lld"; break;
                case LinkerStyle::Mac: llvm::errs() << "ld64.lld"; break;
                case LinkerStyle::Wasm: llvm::errs() << "wasm-lld"; break;
            }
            llvm::errs() << ">";
            for (const auto& iter : arguments)
            {
                llvm::errs() << " " << iter;
            }
            llvm::errs() << '\n';
            if (commandLine.onlyPrint())
            {
                return true;
            }
        }
        std::vector<const char*> refs(1 + arguments.size());
        refs[0] = "pylir";
        std::transform(arguments.begin(), arguments.end(), 1 + refs.begin(),
                       [](const std::string& string) { return string.c_str(); });
        switch (style)
        {
            case LinkerStyle::MSVC: return lld::coff::link(refs, llvm::outs(), llvm::errs(), false, false);
            case LinkerStyle::GNU:
                if (m_triple.isOSCygMing())
                {
                    return lld::mingw::link(refs, llvm::outs(), llvm::errs(), false, false);
                }
                return lld::elf::link(refs, llvm::outs(), llvm::errs(), false, false);
            case LinkerStyle::Mac: return lld::macho::link(refs, llvm::outs(), llvm::errs(), false, false);
            case LinkerStyle::Wasm: return lld::wasm::link(refs, llvm::outs(), llvm::errs(), false, false);
        }
        PYLIR_UNREACHABLE;
    }
#endif
    if (commandLine.verbose() || commandLine.onlyPrint())
    {
        llvm::errs() << linkerPath;
        for (const auto& iter : arguments)
        {
            llvm::errs() << " " << iter;
        }
        if (commandLine.onlyPrint())
        {
            return true;
        }
    }
    std::vector<llvm::StringRef> refs(arguments.begin(), arguments.end());
    return llvm::sys::ExecuteAndWait(linkerPath, refs) == 0;
}

pylir::Toolchain::Stdlib pylir::Toolchain::getStdlib(const pylir::cli::CommandLine& commandLine) const
{
    return llvm::StringSwitch<Stdlib>(commandLine.getArgs().getLastArgValue(pylir::cli::OPT_stdlib_EQ))
        .Case("libc++", Stdlib::Libcpp)
        .Case("libstdc++", Stdlib::Libstdcpp)
        .Default(defaultStdlib());
}

pylir::Toolchain::RTLib pylir::Toolchain::getRTLib(const pylir::cli::CommandLine& commandLine) const
{
    return llvm::StringSwitch<RTLib>(commandLine.getArgs().getLastArgValue(pylir::cli::OPT_rtlib_EQ))
        .Case("compiler-rt", RTLib::CompilerRt)
        .Case("libgcc", RTLib::Libgcc)
        .Default(defaultRTLib());
}

bool pylir::Toolchain::isPIE(const pylir::cli::CommandLine& commandLine) const
{
    return commandLine.getArgs().hasFlag(pylir::cli::OPT_fpie, pylir::cli::OPT_fno_pie, defaultsToPIE());
}

std::vector<std::string> pylir::Toolchain::getLLVMOptions(const llvm::opt::InputArgList& args) const
{
    std::vector<std::string> result;

    // Allow callee saved registers for live-through and GC ptr values
    result.emplace_back("-fixup-allow-gcptr-in-csr");
    if (args.getLastArgValue(pylir::cli::OPT_O, "0") != "0")
    {
        // No restrictions on how many registers its allowed to use
        result.emplace_back("-max-registers-for-gc-values=1000");
    }

    auto options = args.getAllArgValues(pylir::cli::OPT_mllvm);
    result.insert(result.end(), std::move_iterator(options.begin()), std::move_iterator(options.end()));
    return result;
}
