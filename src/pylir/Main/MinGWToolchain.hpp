
#pragma once

#include "Toolchain.hpp"

namespace pylir
{
class MinGWToolchain : public Toolchain
{
public:
    explicit MinGWToolchain(const llvm::Triple& triple);

    bool link(const cli::CommandLine& commandLine, llvm::StringRef objectFile) const override;
};
} // namespace pylir
