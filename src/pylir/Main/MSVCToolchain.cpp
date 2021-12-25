
#include "MSVCToolchain.hpp"

pylir::MSVCToolchain::MSVCToolchain(const llvm::Triple& triple) : Toolchain(triple) {}

bool pylir::MSVCToolchain::link(const pylir::cli::CommandLine& commandLine, llvm::StringRef objectFile) const
{
    return false;
}
