
#pragma once

#include "Toolchain.hpp"

namespace pylir
{
class MSVCToolchain : public Toolchain
{
protected:
    // These values don't actually matter for MSVC toolchains

    [[nodiscard]] Stdlib defaultStdlib() const override
    {
        return Stdlib::libcpp;
    }
    [[nodiscard]] RTLib defaultRTLib() const override
    {
        return RTLib::compiler_rt;
    }

    [[nodiscard]] bool defaultsToPIC() const override;

public:
    explicit MSVCToolchain(const llvm::Triple& triple, const cli::CommandLine& commandLine);

    [[nodiscard]] bool link(const cli::CommandLine& commandLine, llvm::StringRef objectFile) const override;
};
} // namespace pylir
