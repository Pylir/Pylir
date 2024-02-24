//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Toolchain.hpp"

#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Program.h>

#include <lld/Common/Driver.h>

LLD_HAS_DRIVER(coff)
LLD_HAS_DRIVER(mingw)
LLD_HAS_DRIVER(elf)
LLD_HAS_DRIVER(wasm)
LLD_HAS_DRIVER(macho)

#include "DiagnosticMessages.hpp"
#include "Version.hpp"

void pylir::Toolchain::parseSanitizers(cli::CommandLine& commandLine) {
  const llvm::opt::Arg* lastSanArg = nullptr;
  for (const llvm::opt::Arg* iter : commandLine.getArgs()) {
    if (!iter->getOption().matches(cli::OPT_Xsanitize_EQ))
      continue;

    lastSanArg = iter;
    iter->claim();
    for (llvm::StringRef sanitizer : llvm::split(iter->getValue(), ",")) {
      if (sanitizer == "address") {
        m_useAddressSanitizer = true;
      } else if (sanitizer == "thread") {
        m_useThreadSanitizer = true;
      } else if (sanitizer == "undefined") {
        m_useUndefinedSanitizer = true;
      } else {
        static_assert(Diag::hasLocationProvider_v<const llvm::opt::Arg*,
                                                  cli::CommandLine>);

        commandLine.createError(iter, Diag::UNKNOWN_SANITIZER_N, sanitizer)
            .addHighlight(iter);
        return;
      }
    }
  }
  if (!m_useThreadSanitizer || !m_useAddressSanitizer)
    return;

  PYLIR_ASSERT(lastSanArg);
  commandLine
      .createError(
          lastSanArg,
          Diag::ADDRESS_AND_THREAD_SANITIZERS_ARE_INCOMPATIBLE_WITH_EACH_OTHER)
      .addHighlight(lastSanArg);
}

pylir::Toolchain::Toolchain(llvm::Triple triple, cli::CommandLine& commandLine)
    : m_triple(std::move(triple)) {
  parseSanitizers(commandLine);

  // Runtime directory where our runtime libraries are placed.
  llvm::SmallString<10> pylirRuntimeDir = commandLine.getExecutablePath();
  llvm::sys::path::remove_filename(pylirRuntimeDir);
  llvm::sys::path::append(pylirRuntimeDir, "..", "lib", "pylir",
                          m_triple.str());
  m_builtinLibrarySearchDirs.emplace_back(pylirRuntimeDir);

  // Directories where to search for a linker.
  llvm::SmallString<10> executablePath = commandLine.getExecutablePath();
  llvm::sys::path::remove_filename(executablePath);
  m_programPaths.emplace_back(executablePath);
}

bool pylir::Toolchain::callLinker(
    cli::CommandLine& commandLine,
    LinkerInvocationBuilder&& linkerInvocationBuilder) const {
  const auto& args = commandLine.getArgs();
  std::string linkerPath;
  if (auto* arg = args.getLastArg(pylir::cli::OPT_lld_path_EQ)) {
    linkerPath = arg->getValue();
  } else if (!args.hasFlag(pylir::cli::OPT_fintegrated_lld,
                           pylir::cli::OPT_fno_integrated_lld, true)) {
    std::vector<llvm::StringRef> candidates;
    switch (linkerInvocationBuilder.getLinkerStyle()) {
    case LinkerStyle::MSVC: candidates = {"lld-link"}; break;
    case LinkerStyle::MinGW:
    case LinkerStyle::ELF: candidates = {"ld.lld"}; break;
    case LinkerStyle::Mac: candidates = {"ld64.lld"}; break;
    case LinkerStyle::Wasm: candidates = {"wasm-lld"}; break;
    }
    std::vector<std::string> attempts;
    for (auto iter : candidates) {
      std::vector<llvm::StringRef> refs(m_programPaths.begin(),
                                        m_programPaths.end());
      std::string variant = iter.str();
      attempts.push_back(variant);
      auto result = llvm::sys::findProgramByName(variant, refs);
      if (result) {
        linkerPath = std::move(*result);
        break;
      }
      result = llvm::sys::findProgramByName(variant);
      if (result) {
        linkerPath = std::move(*result);
        break;
      }
      variant = (m_triple.str() + "-" + iter).str();
      attempts.push_back(variant);
      result = llvm::sys::findProgramByName(variant, refs);
      if (result) {
        linkerPath = std::move(*result);
        break;
      }
      result = llvm::sys::findProgramByName(variant);
      if (result) {
        linkerPath = std::move(*result);
        break;
      }
    }
    if (linkerPath.empty()) {
      commandLine.createError(pylir::Diag::FAILED_TO_FIND_LINKER)
          .addNote(pylir::Diag::ATTEMPTED_N, fmt::join(attempts, ", "));
      return false;
    }
  }
  if (linkerPath.empty()) {
    if (commandLine.verbose() || commandLine.onlyPrint()) {
      llvm::errs() << "<builtin-";
      switch (linkerInvocationBuilder.getLinkerStyle()) {
      case LinkerStyle::MSVC: llvm::errs() << "lld-link"; break;
      case LinkerStyle::MinGW:
      case LinkerStyle::ELF: llvm::errs() << "ld.lld"; break;
      case LinkerStyle::Mac: llvm::errs() << "ld64.lld"; break;
      case LinkerStyle::Wasm: llvm::errs() << "wasm-lld"; break;
      }
      llvm::errs() << ">";
      for (const auto& iter : linkerInvocationBuilder.getArgs())
        llvm::errs() << " " << iter;

      llvm::errs() << '\n';
      if (commandLine.onlyPrint())
        return true;
    }
    std::vector<const char*> refs(1 + linkerInvocationBuilder.getArgs().size());
    refs[0] = "pylir";
    llvm::transform(linkerInvocationBuilder.getArgs(), 1 + refs.begin(),
                    [](const std::string& string) { return string.c_str(); });
    switch (linkerInvocationBuilder.getLinkerStyle()) {
    case LinkerStyle::MSVC:
      return lld::coff::link(refs, llvm::outs(), llvm::errs(), false, false);
    case LinkerStyle::MinGW:
      return lld::mingw::link(refs, llvm::outs(), llvm::errs(), false, false);
    case LinkerStyle::ELF:
      return lld::elf::link(refs, llvm::outs(), llvm::errs(), false, false);
    case LinkerStyle::Mac:
      return lld::macho::link(refs, llvm::outs(), llvm::errs(), false, false);
    case LinkerStyle::Wasm:
      return lld::wasm::link(refs, llvm::outs(), llvm::errs(), false, false);
    }
    PYLIR_UNREACHABLE;
  }
  if (commandLine.verbose() || commandLine.onlyPrint()) {
    llvm::errs() << linkerPath;
    for (const auto& iter : linkerInvocationBuilder.getArgs())
      llvm::errs() << " " << iter;

    if (commandLine.onlyPrint())
      return true;
  }
  std::vector<llvm::StringRef> refs(linkerInvocationBuilder.getArgs().begin(),
                                    linkerInvocationBuilder.getArgs().end());
  return llvm::sys::ExecuteAndWait(linkerPath, refs) == 0;
}

bool pylir::Toolchain::isPIE(const pylir::cli::CommandLine& commandLine) const {
  return commandLine.getArgs().hasFlag(
      pylir::cli::OPT_fpie, pylir::cli::OPT_fno_pie, defaultsToPIE());
}

std::vector<std::string>
pylir::Toolchain::getLLVMOptions(const llvm::opt::InputArgList& args) const {
  std::vector<std::string> result;
  // Allow callee saved registers for live-through and GC ptr values
  result.emplace_back("-fixup-allow-gcptr-in-csr");
  if (args.getLastArgValue(pylir::cli::OPT_O, "0") != "0") {
    // No restrictions on how many registers its allowed to use
    result.emplace_back("-max-registers-for-gc-values=1000");
  }

  auto options = args.getAllArgValues(pylir::cli::OPT_mllvm);
  result.insert(result.end(), std::move_iterator(options.begin()),
                std::move_iterator(options.end()));
  return result;
}

std::string pylir::Toolchain::findOnBuiltinPaths(llvm::StringRef file) const {
  auto sep = llvm::sys::path::get_separator();
  for (const auto& iter : m_builtinLibrarySearchDirs)
    if (llvm::sys::fs::exists(iter + sep + file))
      return (iter + sep + file).str();

  return file.str();
}

namespace {
llvm::StringRef getOSLibName(const llvm::Triple& triple) {
  if (triple.isOSDarwin())
    return "darwin";

  switch (triple.getOS()) {
  case llvm::Triple::FreeBSD: return "freebsd";
  case llvm::Triple::NetBSD: return "netbsd";
  case llvm::Triple::OpenBSD: return "openbsd";
  case llvm::Triple::Solaris: return "sunos";
  case llvm::Triple::AIX: return "aix";
  default: return triple.getOSName();
  }
}
} // namespace

pylir::ClangInstallation pylir::ClangInstallation::searchForClangInstallation(
    llvm::ArrayRef<std::string> rootDirCandidates, const llvm::Triple& triple) {
  std::string rootDir;
  std::string runtimeDir;
  bool perTargetRuntimeDir = false;

  pylir::Version currentVersion;
  for (const auto& iter : rootDirCandidates) {
    llvm::SmallString<32> path{iter};
    llvm::sys::path::append(path, "clang");
    if (!llvm::sys::fs::exists(path)) {
      path = iter;
      llvm::sys::path::append(path, "lib", "clang");
      if (!llvm::sys::fs::exists(path))
        continue;
    }
    std::error_code ec;
    for (llvm::sys::fs::directory_iterator begin(path, ec), end;
         !ec && begin != end; begin = begin.increment(ec)) {
      auto newVersion =
          pylir::Version::parse(llvm::sys::path::filename(begin->path()));
      if (!newVersion)
        continue;

      if (currentVersion > *newVersion)
        continue;

      llvm::SmallString<32> temp{begin->path()};
      llvm::sys::path::append(temp, "lib", triple.str());
      perTargetRuntimeDir = llvm::sys::fs::exists(temp);
      if (!perTargetRuntimeDir) {
        llvm::sys::path::remove_filename(temp);
        llvm::sys::path::append(temp, getOSLibName(triple));
      }
      currentVersion = std::move(*newVersion);
      rootDir = iter;
      runtimeDir = temp.str();
    }
  }
  return ClangInstallation(perTargetRuntimeDir, std::move(runtimeDir),
                           std::move(rootDir));
}

std::string
pylir::ClangInstallation::getRuntimeLibname(llvm::StringRef name,
                                            const llvm::Triple& triple) const {
  // Darwin platforms are a bit extra special here. The builtins library simply
  // has the 'builtins' component missing.
  std::string libName = "clang_rt";
  if (!triple.isOSDarwin() || name != "builtins")
    libName += '.' + name.str();

  if (m_perTargetRuntimeDir)
    return libName;

  // Darwin platforms also do not use the arch name as suffix, but the OS
  // instead.
  if (!triple.isOSDarwin())
    return libName + '-' + triple.getArchName().str();

  llvm::StringRef suffix;
  if (triple.isMacOSX())
    suffix = "osx";

  return libName + '.' + suffix.str();
}
