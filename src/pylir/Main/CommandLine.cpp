#include "CommandLine.hpp"

#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>

pylir::cli::PylirOptTable::PylirOptTable() : llvm::opt::OptTable(InfoTable) {}

pylir::cli::CommandLine::CommandLine(int argc, char** argv)
    : m_saver(m_allocator),
      m_args(m_table.parseArgs(argc, argv, OPT_UNKNOWN, m_saver,
                               [this](llvm::StringRef msg)
                               {
                                   m_errorsOccurred = true;
                                   llvm::errs() << pylir::Diag::formatLine(pylir::Diag::Severity::Error, msg);
                               })),
      m_exe(argv[0]),
      m_rendered(
          [this]
          {
              std::string rendered = llvm::sys::path::filename(m_exe).str();
              for (auto iter : m_args)
              {
                  auto arg = iter->getAsString(m_args);
                  rendered += " ";
                  m_argRanges.insert({iter, {rendered.size(), rendered.size() + arg.size()}});
                  rendered += arg;
              }
              return rendered;
          }(),
          "<command-line>")
{
}

void pylir::cli::CommandLine::printHelp(llvm::raw_ostream& out) const
{
    m_table.printHelp(out, (llvm::sys::path::filename(m_exe) + " [options] <input>").str().c_str(),
                      "Optimizing Python compiler using MLIR and LLVM");
}

void pylir::cli::CommandLine::printVersion(llvm::raw_ostream& out) const
{
    out << "pylir " PYLIR_VERSION "\n";
}
