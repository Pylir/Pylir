#pragma once

#include <llvm/ADT/Triple.h>

#include "CommandLine.hpp"

namespace pylir
{
class Toolchain
{
    llvm::Triple m_triple;

public:
    explicit Toolchain(llvm::Triple triple) : m_triple(std::move(triple)) {}

    virtual ~Toolchain() = default;

    virtual bool link(const pylir::cli::CommandLine& commandLine, llvm::StringRef objectFile) const = 0;
};
} // namespace pylir
