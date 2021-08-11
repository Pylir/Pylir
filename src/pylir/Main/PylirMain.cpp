
#include "PylirMain.hpp"

#include <llvm/Option/Arg.h>
#include <llvm/Option/ArgList.h>
#include <llvm/Option/Option.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/StringSaver.h>
#include <llvm/Support/raw_ostream.h>

#include <pylir/Main/Opts.inc>

namespace
{
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

class PylirOptTable : public llvm::opt::OptTable
{
public:
    PylirOptTable() : OptTable(InfoTable) {}
};
} // namespace

int pylir::main(int argc, char* argv[])
{
    llvm::BumpPtrAllocator a;
    llvm::StringSaver saver(a);
    PylirOptTable table;
    auto args = table.parseArgs(argc, argv, OPT_UNKNOWN, saver,
                                [&](llvm::StringRef msg)
                                {
                                    // TODO:
                                    llvm::errs() << msg;
                                    std::terminate();
                                });

    if (args.hasArg(OPT_help))
    {
        table.printHelp(llvm::outs(), (llvm::Twine(argv[0]) + " [options] <input files>").str().c_str(),
                        "Python optimizing MLIR compiler");
        return 0;
    }

    return 0;
}
