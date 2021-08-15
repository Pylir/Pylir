
#pragma once

#include <llvm/Option/ArgList.h>
#include <llvm/Option/OptTable.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/StringSaver.h>

#include <pylir/Diagnostics/DiagnosticsBuilder.hpp>
#include <pylir/Diagnostics/Document.hpp>

namespace pylir
{
namespace cli
{
class PylirOptTable : public llvm::opt::OptTable
{
public:
    PylirOptTable();
};

class CommandLine
{
    llvm::BumpPtrAllocator m_allocator;
    llvm::StringSaver m_saver;
    PylirOptTable m_table;
    llvm::opt::InputArgList m_args;
    bool m_errorsOccurred = false;
    std::string m_exe;
    llvm::DenseMap<llvm::opt::Arg*, std::pair<std::size_t, std::size_t>> m_argRanges;
    pylir::Diag::Document m_rendered;

    friend struct pylir::Diag::LocationProvider<llvm::opt::Arg*, void>;

public:
    explicit CommandLine(int argc, char* argv[]);

    explicit operator bool() const
    {
        return !m_errorsOccurred;
    }

    const llvm::opt::InputArgList& getArgs() const
    {
        return m_args;
    }

    void printHelp(llvm::raw_ostream& out) const;

    void printVersion(llvm::raw_ostream& out) const;

    template <class T, class S, class... Args>
    [[nodiscard]] pylir::Diag::DiagnosticsBuilder createDiagnosticsBuilder(const T& location, const S& message,
                                                                           Args&&... args) const
    {
        return pylir::Diag::DiagnosticsBuilder(this, m_rendered, location, message, std::forward<Args>(args)...);
    }
};

} // namespace cli

int main(int argc, char* argv[]);
} // namespace pylir
