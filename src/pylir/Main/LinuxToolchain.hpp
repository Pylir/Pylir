
#pragma once

#include "Toolchain.hpp"

namespace pylir
{
class LinuxToolchain : public Toolchain
{
public:
    explicit LinuxToolchain(const llvm::Triple& triple);

    bool link(const cli::CommandLine& commandLine, llvm::StringRef objectFile) const override;
};
} // namespace pylir
