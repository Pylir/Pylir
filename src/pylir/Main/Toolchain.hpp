#pragma once

#include <llvm/ADT/Triple.h>

#include "CommandLine.hpp"

namespace pylir
{
class Toolchain
{
protected:
    llvm::Triple m_triple;
    std::vector<std::string> m_programPaths;

    enum class LinkerStyle
    {
        MSVC,
        GNU,
        Mac,
        Wasm,
    };

    bool callLinker(const pylir::cli::CommandLine& commandLine, LinkerStyle style,
                    llvm::ArrayRef<std::string> arguments) const;

    enum class Stdlib
    {
        libstdcpp,
        libcpp,
    };

    virtual Stdlib defaultStdlib() const = 0;

    Stdlib getStdlib(const pylir::cli::CommandLine& commandLine) const;

    enum class RTLib
    {
        compiler_rt,
        libgcc,
    };

    virtual RTLib defaultRTLib() const = 0;

    RTLib getRTLib(const pylir::cli::CommandLine& commandLine) const;

    virtual bool defaultsToPIE() const
    {
        return false;
    }

public:
    explicit Toolchain(llvm::Triple triple) : m_triple(std::move(triple)) {}

    virtual ~Toolchain() = default;

    virtual bool link(const pylir::cli::CommandLine& commandLine, llvm::StringRef objectFile) const = 0;

    bool isPIE(const pylir::cli::CommandLine& commandLine) const;

    virtual bool defaultsToPIC() const
    {
        return false;
    }
};
} // namespace pylir
