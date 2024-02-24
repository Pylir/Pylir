//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/Option/ArgList.h>
#include <llvm/Option/OptTable.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/StringSaver.h>

#include <pylir/Diagnostics/DiagnosticsBuilder.hpp>
#include <pylir/Diagnostics/DiagnosticsManager.hpp>
#include <pylir/Diagnostics/Document.hpp>

namespace pylir::cli {

class PylirOptTable : public llvm::opt::PrecomputedOptTable {
public:
  PylirOptTable();
};

class CommandLine {
  pylir::Diag::DiagnosticsNoDocManager m_locLessDM;
  llvm::BumpPtrAllocator m_allocator;
  llvm::StringSaver m_saver;
  PylirOptTable m_table;
  llvm::opt::InputArgList m_args;
  std::string m_exe;
  llvm::DenseMap<const llvm::opt::Arg*, std::pair<std::size_t, std::size_t>>
      m_argRanges;
  pylir::Diag::Document m_rendered;
  pylir::Diag::DiagnosticsDocManager<CommandLine> m_commandLineDM;

  friend struct pylir::Diag::LocationProvider<const llvm::opt::Arg*>;

public:
  explicit CommandLine(std::string exe, int argc, char** argv,
                       pylir::Diag::DiagnosticsManager& diagnosticsManager);

  explicit operator bool() const {
    return !m_locLessDM.errorsOccurred() && !m_commandLineDM.errorsOccurred();
  }

  const llvm::opt::InputArgList& getArgs() const {
    return m_args;
  }

  llvm::StringRef getExecutablePath() const {
    return m_exe;
  }

  bool verbose() const;

  bool onlyPrint() const;

  void printHelp(llvm::raw_ostream& out) const;

  void printVersion(llvm::raw_ostream& out) const;

  template <
      class T, class S, class... Args,
      std::enable_if_t<Diag::hasLocationProvider_v<T, CommandLine>>* = nullptr>
  auto createError(const T& location, const S& message, Args&&... args) {
    return Diag::DiagnosticsBuilder(m_commandLineDM, Diag::Severity::Error,
                                    location, message,
                                    std::forward<Args>(args)...);
  }

  template <
      class T, class S, class... Args,
      std::enable_if_t<Diag::hasLocationProvider_v<T, CommandLine>>* = nullptr>
  auto createWarning(const T& location, const S& message, Args&&... args) {
    return Diag::DiagnosticsBuilder(m_commandLineDM, Diag::Severity::Warning,
                                    location, message,
                                    std::forward<Args>(args)...);
  }

  template <class S, class... Args>
  auto createError(const S& message, Args&&... args) {
    return Diag::DiagnosticsBuilder(m_locLessDM, Diag::Severity::Error, message,
                                    std::forward<Args>(args)...);
  }

  template <class S, class... Args>
  auto createWarning(const S& message, Args&&... args) {
    return Diag::DiagnosticsBuilder(m_locLessDM, Diag::Severity::Warning,
                                    message, std::forward<Args>(args)...);
  }
};

enum ID {
  // NOLINTNEXTLINE(readability-identifier-naming): Predefined name.
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include <pylir/Main/Opts.inc>
#undef OPTION
};

} // namespace pylir::cli

namespace pylir::Diag {
template <>
struct LocationProvider<const llvm::opt::Arg*> {
  static std::pair<std::size_t, std::size_t>
  getRange(const llvm::opt::Arg* value,
           const cli::CommandLine& commandLine) noexcept {
    return commandLine.m_argRanges.lookup(value);
  }
};

template <>
struct LocationProvider<llvm::opt::Arg*>
    : LocationProvider<const llvm::opt::Arg*> {};
} // namespace pylir::Diag
