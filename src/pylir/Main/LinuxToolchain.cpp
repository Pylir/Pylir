//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "LinuxToolchain.hpp"

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Program.h>

#include "Distro.hpp"
#include "Version.hpp"

namespace {

struct GCCInstallation {
  std::string libPath;
  std::string gccLibPath;
  llvm::Triple gccTriple;
};

void collectLibDirsAndTriples(const llvm::Triple& triple,
                              llvm::SmallVectorImpl<llvm::StringRef>& libDirs,
                              llvm::SmallVectorImpl<llvm::StringRef>& triples) {
  libDirs.emplace_back("lib");
  if (triple.isArch64Bit())
    libDirs.emplace_back("lib64");
  else if (triple.isArch32Bit())
    libDirs.emplace_back("lib32");

  triples.emplace_back(triple.str());
  switch (triple.getArch()) {
  case llvm::Triple::x86_64: {
    triples.append(
        {"x86_64-linux-gnu", "x86_64-unknown-linux-gnu", "x86_64-pc-linux-gnu",
         "x86_64-redhat-linux6E", "x86_64-redhat-linux", "x86_64-suse-linux",
         "x86_64-manbo-linux-gnu", "x86_64-linux-gnu", "x86_64-slackware-linux",
         "x86_64-unknown-linux", "x86_64-amazon-linux"});
    break;
  }
  default: break;
  }
}

std::optional<GCCInstallation>
findGCCInstallation(const llvm::Triple& triple,
                    const pylir::cli::CommandLine& commandLine) {
  llvm::SmallVector<llvm::StringRef, 4> candidateLibDirs;
  llvm::SmallVector<llvm::StringRef, 16> candidateTriples;
  collectLibDirsAndTriples(triple, candidateLibDirs, candidateTriples);

  llvm::SmallVector<std::string, 8> prefixes;
  // Always respect the users wish of the sysroot if present.
  if (auto* arg =
          commandLine.getArgs().getLastArg(pylir::cli::OPT_sysroot_EQ)) {
    llvm::SmallString<32> temp;
    llvm::sys::path::append(temp, arg->getValue());
    prefixes.emplace_back(temp);
    llvm::sys::path::append(temp, "usr");
    prefixes.emplace_back(temp);
  } else {
    // Otherwise, we check two default locations: The system sysroot and the
    // sysroot in the compiler installation.
    llvm::SmallString<32> executablePath = commandLine.getExecutablePath();
    llvm::sys::path::remove_filename(executablePath);
    llvm::sys::path::append(executablePath, "..", triple.str());
    prefixes.emplace_back(executablePath);
    llvm::sys::path::append(executablePath, "usr");
    prefixes.emplace_back(executablePath);
    prefixes.push_back("/");
    prefixes.push_back("/usr");
  }

  pylir::Version currentVersion;
  std::string libPath;
  std::string gccLibPath;
  llvm::Triple resultTriple;
  for (auto& prefix : prefixes) {
    if (!llvm::sys::fs::exists(prefix))
      continue;

    for (auto suffix : candidateLibDirs) {
      for (auto gccTriple : candidateTriples) {
        llvm::SmallString<32> path{prefix};
        llvm::sys::path::append(path, suffix, "gcc", gccTriple);
        if (!llvm::sys::fs::exists(path))
          continue;

        std::error_code ec;
        for (llvm::sys::fs::directory_iterator iter(path, ec), end;
             !ec && iter != end; iter = iter.increment(ec)) {
          auto newVersion =
              pylir::Version::parse(llvm::sys::path::filename(iter->path()));
          if (!newVersion)
            continue;

          if (currentVersion < *newVersion) {
            currentVersion = std::move(*newVersion);
            llvm::SmallString<32> temp(prefix);
            llvm::sys::path::append(temp, suffix);
            libPath = temp.str();
            gccLibPath = iter->path();
            resultTriple = llvm::Triple(gccTriple);
          }
        }
      }
    }
  }
  if (!currentVersion)
    return {};

  return GCCInstallation{std::move(libPath), std::move(gccLibPath),
                         std::move(resultTriple)};
}

const char* getDynamicLinker(const llvm::Triple& triple,
                             const pylir::cli::CommandLine&) {
  switch (triple.getArch()) {
  case llvm::Triple::x86_64: return "/lib64/ld-linux-x86-64.so.2";
  default: return nullptr;
  }
}

} // namespace

void pylir::LinuxToolchain::findClangInstallation(
    const cli::CommandLine& commandLine) {
  (void)commandLine;
  std::vector<std::string> candidates;
#ifdef __linux__
  if (!commandLine.getArgs().hasArg(cli::OPT_sysroot_EQ)) {
    // If on a linux system without the user having specified a custom sysroot
    // we'll also add the path of clang++ as candidate, hoping that it will lead
    // to its install directory.
    if (auto result = llvm::sys::findProgramByName("clang++")) {
      llvm::SmallString<32> temp;
      if (!llvm::sys::fs::real_path(*result, temp)) {
        // Strip filename and get out of bin.
        candidates.emplace_back(
            llvm::sys::path::parent_path(llvm::sys::path::parent_path(temp)));
      }
    }
  }
#endif
  llvm::append_range(candidates, m_builtinLibrarySearchDirs);
  // Some distros put the clang resource directories within their LLVM install
  // directories, hence we'll just add any LLVM installs in the sysroot as
  // candidates.
  for (llvm::StringRef dir : m_builtinLibrarySearchDirs) {
    std::error_code ec;
    for (llvm::sys::fs::directory_iterator iter(dir, ec), end;
         !ec && iter != end; iter = iter.increment(ec)) {
      if (iter->type() != llvm::sys::fs::file_type::directory_file)
        continue;

      llvm::StringRef path = iter->path();
      llvm::StringRef fileName = llvm::sys::path::filename(path);
      if (fileName.starts_with("llvm"))
        candidates.emplace_back(path);
    }
  }

  m_clangInstallation =
      ClangInstallation::searchForClangInstallation(candidates, m_triple);
}

pylir::LinuxToolchain::LinuxToolchain(llvm::Triple triple,
                                      cli::CommandLine& commandLine)
    : Toolchain(std::move(triple), commandLine) {
  auto gccInstall = findGCCInstallation(m_triple, commandLine);
  if (!gccInstall)
    return;

  m_sysroot = llvm::sys::path::parent_path(gccInstall->libPath);
  if (llvm::sys::path::filename(m_sysroot) == "usr")
    m_sysroot = llvm::sys::path::parent_path(m_sysroot);

  m_builtinLibrarySearchDirs.emplace_back(gccInstall->libPath);
  m_builtinLibrarySearchDirs.emplace_back(gccInstall->gccLibPath);

  if (m_triple.isArch64Bit()) {
    addIfExists(gccInstall->libPath, "..", "lib64");
    addIfExists(gccInstall->libPath, "..", gccInstall->gccTriple.str(),
                "lib64");
  }
  addIfExists(m_sysroot, "lib", gccInstall->gccTriple.str());
  addIfExists(m_sysroot, "usr", "lib", gccInstall->gccTriple.str());

  // Clang installation is not required on Linux except for sanitizers.
  if (useSanitizers())
    findClangInstallation(commandLine);
}

bool pylir::LinuxToolchain::link(cli::CommandLine& commandLine,
                                 llvm::StringRef objectFile) const {
  const auto& args = commandLine.getArgs();

  auto linkerInvocation = LinkerInvocationBuilder(LinkerStyle::ELF);
  linkerInvocation.addArg("--sysroot=" + m_sysroot, !m_sysroot.empty())
      .addArg("-pie", isPIE(commandLine));

  Distro distro(m_triple);
  if (distro.isAlpineLinux())
    linkerInvocation.addArgs("-z", "now");

  if (distro.isOpenSuse() || distro.isUbuntu() || distro.isAlpineLinux())
    linkerInvocation.addArgs("-z", "relro");

  if (distro.isRedhat() || distro.isOpenSuse() || distro.isAlpineLinux() ||
      (distro.isUbuntu() && distro >= Distro::UbuntuMaverick))
    linkerInvocation.addArg("--hash-style=gnu");

  if (distro.isDebian() || distro.isOpenSuse() ||
      distro == Distro::UbuntuLucid || distro == Distro::UbuntuJaunty ||
      distro == Distro::UbuntuKarmic)
    linkerInvocation.addArg("--hash-style=both");

  linkerInvocation.addArg("--enable-new-dtags", distro.isOpenSuse())
      .addArg("--eh-frame-hdr")
      .addEmulation(m_triple)
      .addArgs("-dynamic-linker", getDynamicLinker(m_triple, commandLine))
      .addLLVMOptions(getLLVMOptions(args));

  if (auto* output = args.getLastArg(cli::OPT_o))
    linkerInvocation.addOutputFile(output->getValue());
  else if (auto* input = args.getLastArg(cli::OPT_INPUT))
    linkerInvocation.addOutputFile(llvm::sys::path::stem(input->getValue()));

  linkerInvocation.addArg(findOnBuiltinPaths("crt1.o"))
      .addArg(findOnBuiltinPaths("crti.o"))
      .addArg(findOnBuiltinPaths("crtbegin.o"))
      .addLibrarySearchDirs(args.getAllArgValues(cli::OPT_L))
      .addLibrarySearchDirs(m_builtinLibrarySearchDirs);

  bool needsExportDynamic = useSanitizers();

  auto addSanitizer = [&](llvm::StringRef name) {
    // We need to always link the static sanitizers. The shared ones are only
    // used when linking a shared library.
    llvm::SmallString<64> temp = m_clangInstallation.getRuntimeDir();
    llvm::sys::path::append(
        temp,
        "lib" + m_clangInstallation.getRuntimeLibname(name, m_triple) + ".a");
    if (!llvm::sys::fs::exists(temp))
      return;

    linkerInvocation.addArg("--whole-archive")
        .addArg(temp)
        .addArg("--no-whole-archive");
    if (llvm::sys::fs::exists(temp + ".syms")) {
      linkerInvocation.addArg("--dynamic-list=" + temp + ".syms");
      needsExportDynamic = false;
    }
  };

  if (useAddressSanitizer()) {
    addSanitizer("asan_static");
    addSanitizer("asan");
    addSanitizer("asan_cxx");
  }
  if (useThreadSanitizer()) {
    addSanitizer("tsan");
    addSanitizer("tsan_cxx");
  }
  if (useUndefinedSanitizer()) {
    addSanitizer("ubsan_standalone");
    addSanitizer("ubsan_standalone_cxx");
  }

  if (useSanitizers()) {
    // The sanitizers have these as link dependencies, which we currently do not
    // otherwise.
    linkerInvocation.addLibrary("pthread");
    linkerInvocation.addLibrary("dl");
  }

  if (needsExportDynamic)
    linkerInvocation.addArg("--export-dynamic");

  // Make sure the order of -l and -Wl are preserved.
  for (auto* arg : args) {
    if (arg->getOption().matches(cli::OPT_l)) {
      linkerInvocation.addLibrary(arg->getValue());
      continue;
    }
    if (arg->getOption().matches(cli::OPT_Wl)) {
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
      .addArg(findOnBuiltinPaths("crtend.o"))
      .addArg(findOnBuiltinPaths("crtn.o"));

  return callLinker(commandLine, std::move(linkerInvocation));
}
