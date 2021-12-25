
#pragma once

#include <llvm/Option/ArgList.h>
#include <llvm/Option/OptTable.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/StringSaver.h>

#include <pylir/Diagnostics/DiagnosticsBuilder.hpp>
#include <pylir/Diagnostics/Document.hpp>

namespace pylir::cli
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

enum ID
{
    OPT_INVALID = 0, // This is not an option ID.
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM, HELPTEXT, METAVAR, VALUES) OPT_##ID,
#include <pylir/Main/Opts.inc>
#undef OPTION
};

#define PREFIX(NAME, VALUE) const char* const NAME[] = VALUE;
#include <pylir/Main/Opts.inc>
#undef PREFIX

static const llvm::opt::OptTable::Info InfoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM, HELPTEXT, METAVAR, VALUES) \
    {PREFIX, NAME,  HELPTEXT,    METAVAR,     OPT_##ID,  llvm::opt::Option::KIND##Class,                 \
     PARAM,  FLAGS, OPT_##GROUP, OPT_##ALIAS, ALIASARGS, VALUES},
#include <pylir/Main/Opts.inc>
#undef OPTION
};

} // namespace pylir::cli

namespace pylir::Diag
{
template <>
struct LocationProvider<llvm::opt::Arg*, void>
{
    static std::pair<std::size_t, std::size_t> getRange(llvm::opt::Arg* value, const void* context) noexcept
    {
        auto* commandLine = reinterpret_cast<const pylir::cli::CommandLine*>(context);
        return commandLine->m_argRanges.lookup(value);
    }
};
} // namespace pylir::Diag
