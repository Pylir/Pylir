
#include "MinGWToolchain.hpp"

pylir::MinGWToolchain::MinGWToolchain(const llvm::Triple& triple) : Toolchain(triple) {}

bool pylir::MinGWToolchain::link(const pylir::cli::CommandLine& commandLine, llvm::StringRef objectFile) const
{
    return false;
}
