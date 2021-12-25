
#include "LinuxToolchain.hpp"

pylir::LinuxToolchain::LinuxToolchain(const llvm::Triple& triple) : Toolchain(triple) {}

bool pylir::LinuxToolchain::link(const pylir::cli::CommandLine& commandLine, llvm::StringRef objectFile) const
{
    return false;
}
