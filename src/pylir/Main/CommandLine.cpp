//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "CommandLine.hpp"

#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>

#include <utility>

#define PREFIX(NAME, VALUE)                                   \
  static constexpr llvm::StringLiteral NAME##_init[] = VALUE; \
  static constexpr llvm::ArrayRef<llvm::StringLiteral> NAME(  \
      NAME##_init, std::size(NAME##_init) - 1);
#include <pylir/Main/Opts.inc>
#undef PREFIX

static constexpr const llvm::StringLiteral PREFIX_TABLE_INIT[] =
#define PREFIX_UNION(VALUES) VALUES
#include <pylir/Main/Opts.inc>
#undef PREFIX_UNION
    ;
static constexpr const llvm::ArrayRef<llvm::StringLiteral>
    PREFIX_TABLE(PREFIX_TABLE_INIT, std::size(PREFIX_TABLE_INIT) - 1);

// Don't have much choice until this is fixed in LLVM
using llvm::opt::DefaultVis;
using llvm::opt::HelpHidden;
using namespace pylir::cli;

static constexpr llvm::opt::OptTable::Info INFO_TABLE[] = {
#define OPTION(...) LLVM_CONSTRUCT_OPT_INFO(__VA_ARGS__),
#include <pylir/Main/Opts.inc>
#undef OPTION
};

pylir::cli::PylirOptTable::PylirOptTable()
    : llvm::opt::PrecomputedOptTable(INFO_TABLE, PREFIX_TABLE) {}

pylir::cli::CommandLine::CommandLine(
    std::string exe, int argc, char** argv,
    pylir::Diag::DiagnosticsManager& diagnosticsManager)
    : m_locLessDM(diagnosticsManager.createSubDiagnosticManager()),
      m_saver(m_allocator),
      m_args(m_table.parseArgs(argc, argv, OPT_UNKNOWN, m_saver,
                               [this](llvm::StringRef msg) {
                                 Diag::DiagnosticsBuilder(
                                     m_locLessDM, Diag::Severity::Error,
                                     std::string_view{msg});
                               })),
      m_exe(std::move(exe)),
      m_rendered(
          [this] {
            std::string rendered = llvm::sys::path::filename(m_exe).str();
            for (auto* iter : m_args) {
              auto arg = iter->getAsString(m_args);
              rendered += " ";
              m_argRanges.insert(
                  {iter, {rendered.size(), rendered.size() + arg.size()}});
              rendered += arg;
            }
            return rendered;
          }(),
          "<command-line>"),
      m_commandLineDM(
          diagnosticsManager.createSubDiagnosticManager(m_rendered, *this)) {}

void pylir::cli::CommandLine::printHelp(llvm::raw_ostream& out) const {
  m_table.printHelp(
      out,
      (llvm::sys::path::filename(m_exe) + " [options] <input>").str().c_str(),
      "Optimizing Python compiler using MLIR and LLVM");
}

void pylir::cli::CommandLine::printVersion(llvm::raw_ostream& out) const {
  out << "pylir " PYLIR_VERSION "\n";
  out << "LLVM " LLVM_VERSION_STRING "\n";
  out << "MLIR " LLVM_VERSION_STRING "\n";
  out << "lld " LLVM_VERSION_STRING "\n";
}

bool pylir::cli::CommandLine::verbose() const {
  return m_args.hasArg(OPT_verbose);
}

bool pylir::cli::CommandLine::onlyPrint() const {
  return m_args.hasArg(OPT__HASH_HASH_HASH);
}
