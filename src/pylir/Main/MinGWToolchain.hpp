
#pragma once

#include "Toolchain.hpp"

namespace pylir
{
class MinGWToolchain : public Toolchain
{
    std::string m_sysroot;
    std::string m_subdir;

protected:
    Stdlib defaultStdlib() const override;

    RTLib defaultRTLib() const override;

    bool defaultsToPIC() const override;

public:
    explicit MinGWToolchain(const llvm::Triple& triple, const cli::CommandLine& commandLine);

    bool link(const cli::CommandLine& commandLine, llvm::StringRef objectFile) const override;
};
} // namespace pylir
