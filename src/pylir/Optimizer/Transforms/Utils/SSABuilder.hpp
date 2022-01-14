#pragma once

#include <mlir/IR/Block.h>

#include <pylir/Support/Macros.hpp>

#include <functional>

namespace pylir
{

class SSABuilder
{
public:
    using DefinitionsMap = llvm::DenseMap<mlir::Block*, mlir::Value>;

private:
    llvm::DenseMap<mlir::Block*, std::vector<DefinitionsMap*>> m_openBlocks;
    std::function<mlir::Value(mlir::BlockArgument)> m_undefinedCallback;

    mlir::Value tryRemoveTrivialBlockArgument(mlir::BlockArgument argument);

    mlir::Value addBlockArguments(DefinitionsMap& map, mlir::BlockArgument argument);

    mlir::Value readVariableRecursive(mlir::Type type, DefinitionsMap& map, mlir::Block* block);

    void removeBlockArgumentOperands(mlir::BlockArgument argument);

public:
    explicit SSABuilder(std::function<mlir::Value(mlir::BlockArgument)> undefinedCallback = [](auto) -> mlir::Value
                        { PYLIR_UNREACHABLE; })
        : m_undefinedCallback(std::move(undefinedCallback))
    {
    }

    ~SSABuilder()
    {
        PYLIR_ASSERT(m_openBlocks.empty());
    }

    SSABuilder(const SSABuilder&) = delete;
    SSABuilder& operator=(const SSABuilder&) = delete;
    SSABuilder(SSABuilder&&) noexcept = default;
    SSABuilder& operator=(SSABuilder&&) noexcept = default;

    void markOpenBlock(mlir::Block* block);

    void sealBlock(mlir::Block* block);

    mlir::Value readVariable(mlir::Type type, DefinitionsMap& map, mlir::Block* block);
};
} // namespace pylir
