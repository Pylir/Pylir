
#pragma once

#include "Toolchain.hpp"

namespace pylir
{
class MSVCToolchain : public Toolchain
{
public:
    explicit MSVCToolchain(const llvm::Triple& triple);

    bool link(const cli::CommandLine& commandLine, llvm::StringRef objectFile) const override;
};
} // namespace pylir
