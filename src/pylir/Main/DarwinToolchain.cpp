// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "DarwinToolchain.hpp"

#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Program.h>

namespace {
std::unique_ptr<llvm::MemoryBuffer>
getXCRunOutput(std::vector<llvm::StringRef> arguments) {
  llvm::SmallString<64> outputFile;
  llvm::sys::fs::createTemporaryFile("xcrun-output", "", outputFile);

  std::optional<llvm::StringRef> redirects[] = {{""}, outputFile.str(), {""}};

  arguments.insert(arguments.begin(), "/usr/bin/xcrun");
  auto result = llvm::sys::ExecuteAndWait("/usr/bin/xcrun", arguments,
                                          std::nullopt, redirects, 0, 0);
  if (result != 0)
    return nullptr;

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> outputBuffer =
      llvm::MemoryBuffer::getFile(outputFile, true);
  if (!outputBuffer)
    return nullptr;

  return std::move(*outputBuffer);
}
} // namespace

void pylir::DarwinToolchain::deduceSDKRoot(
    const pylir::cli::CommandLine& commandLine) {
  const auto& args = commandLine.getArgs();

  // We need the SDK path where the system libraries are contained.
  // First, respect the users wish if specified via sysroot.
  if (auto* arg = args.getLastArg(cli::OPT_sysroot_EQ)) {
    m_sdkRoot = arg->getValue();
    return;
  }

  // Otherwise we might be running in a xcode developer environment that defines
  // SDKROOT.
  if (char* env = std::getenv("SDKROOT")) {
    if (llvm::sys::path::is_absolute(env) && llvm::sys::fs::exists(env) &&
        llvm::StringRef(env) != "/") {
      m_sdkRoot = env;
      return;
    }
  }

  // As a last resort we attempt to run 'xcrun --show-sdk-path' to get the path.

  std::optional<llvm::StringRef> sdkName;
  switch (m_triple.getOS()) {
  case llvm::Triple::Darwin:
  case llvm::Triple::MacOSX: sdkName = "macosx"; break;
  default: break;
  }
  if (!sdkName)
    return;

  auto outputBuffer = getXCRunOutput({"--sdk", *sdkName, "--show-sdk-path"});
  if (!outputBuffer)
    return;

  m_sdkRoot = outputBuffer->getBuffer().trim();
}

void pylir::DarwinToolchain::searchForClangInstallation() {
  std::vector<std::string> clangInstallationCandidates;
  if (!m_sdkRoot.empty())
    clangInstallationCandidates.push_back(m_sdkRoot);

#ifdef __APPLE__

  if (auto result = llvm::sys::findProgramByName("clang")) {
    // Strip filename and get out of bin.
    clangInstallationCandidates.emplace_back(
        llvm::sys::path::parent_path(llvm::sys::path::parent_path(*result)));
  }

  if (auto appleClangInstall = getXCRunOutput({"--find", "clang"})) {
    // Strip filename and get out of bin.
    clangInstallationCandidates.emplace_back(llvm::sys::path::parent_path(
        llvm::sys::path::parent_path(appleClangInstall->getBuffer().trim())));
  }

#endif

  m_clangInstallation = ClangInstallation::searchForClangInstallation(
      clangInstallationCandidates, m_triple);
}

bool pylir::DarwinToolchain::readSDKSettings(llvm::MemoryBuffer& buffer) {
  auto json = llvm::json::parse(buffer.getBuffer());
  if (!json) {
    llvm::consumeError(json.takeError());
    return false;
  }

  auto* obj = json->getAsObject();
  if (!obj)
    return false;

  auto readVersion =
      [](const llvm::json::Object& object,
         llvm::StringRef key) -> std::optional<llvm::VersionTuple> {
    auto val = object.getString(key);
    if (!val)
      return std::nullopt;

    llvm::VersionTuple version;
    if (version.tryParse(*val))
      return std::nullopt;

    return version;
  };

  m_sdkVersion = readVersion(*obj, "Version");
  return m_sdkVersion.has_value();
}

pylir::DarwinToolchain::DarwinToolchain(llvm::Triple triple,
                                        pylir::cli::CommandLine& commandLine)
    : Toolchain(std::move(triple), commandLine) {
  deduceSDKRoot(commandLine);
  searchForClangInstallation();
  addIfExists(m_clangInstallation.getRuntimeDir());
  if (m_sdkRoot.empty())
    return;

  llvm::SmallString<256> path(m_sdkRoot);
  llvm::sys::path::append(path, "SDKSettings.json");
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> file =
      llvm::MemoryBuffer::getFile(path, true);
  if (!file)
    return;

  if (!readSDKSettings(**file)) {
    // TODO: emit warning with 'commandLine' SDKSettings.json is invalid.
  }
}

namespace {
llvm::StringRef getMachOArchName(const llvm::Triple& triple) {
  switch (triple.getArch()) {
  case llvm::Triple::aarch64_32: return "arm64_32";
  case llvm::Triple::aarch64:
    if (triple.isArm64e())
      return "arm64e";

    return "arm64";
  default: return triple.getArchName();
  }
}
} // namespace

bool pylir::DarwinToolchain::link(pylir::cli::CommandLine& commandLine,
                                  llvm::StringRef objectFile) const {
  const auto& args = commandLine.getArgs();

  auto linkerInvocation = LinkerInvocationBuilder(LinkerStyle::Mac);

  llvm::VersionTuple targetVersion;
  m_triple.getMacOSXVersion(targetVersion);
  llvm::VersionTuple minTargetVersion = m_triple.getMinimumSupportedOSVersion();
  if (!minTargetVersion.empty() && minTargetVersion > targetVersion)
    targetVersion = minTargetVersion;

  llvm::VersionTuple sdkVersion = targetVersion;
  if (m_sdkVersion) {
    sdkVersion = *m_sdkVersion;
    if (!sdkVersion.getMinor())
      sdkVersion = llvm::VersionTuple(sdkVersion.getMajor(), 0);
  }

  linkerInvocation.addArg("-dynamic")
      .addArg("-arch")
      .addArg(getMachOArchName(m_triple))
      .addArg("-platform_version")
      .addArg("macos")
      .addArg(targetVersion.getAsString())
      .addArg(sdkVersion.getAsString())
      .addArg("-pie", isPIE(commandLine))
      .addLLVMOptions(getLLVMOptions(args))
      .addArg("-syslibroot", !m_sdkRoot.empty())
      .addArg(m_sdkRoot, !m_sdkRoot.empty());

  if (auto* output = args.getLastArg(cli::OPT_o))
    linkerInvocation.addOutputFile(output->getValue());
  else if (auto* input = args.getLastArg(cli::OPT_INPUT))
    linkerInvocation.addOutputFile(llvm::sys::path::stem(input->getValue()));

  linkerInvocation.addLibrarySearchDirs(args.getAllArgValues(cli::OPT_L))
      .addLibrarySearchDirs(m_builtinLibrarySearchDirs)
#ifdef __APPLE__
      .addLibrarySearchDir("/", "usr", "local", "lib")
#endif
      ;

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
      .addLibrary("PylirRuntime")
      .addLibrary("PylirMarkAndSweep")
      .addLibrary("PylirRuntimeMain")
      .addLibrary("c++")
      .addLibrary("System")
      .addLibrary(m_clangInstallation.getRuntimeLibname("builtins", m_triple));

  return callLinker(commandLine, std::move(linkerInvocation));
}

bool pylir::DarwinToolchain::defaultsToPIC() const {
  return true;
}
