// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/Block.h>

#include <pylir/Support/Macros.hpp>

#include <functional>

#include "ValueTracker.hpp"

namespace pylir
{

class SSABuilder
{
public:
    using DefinitionsMap = llvm::DenseMap<mlir::Block*, ValueTracker>;

private:
    llvm::DenseMap<mlir::Block*, std::vector<DefinitionsMap*>> m_openBlocks;
    std::function<mlir::Value(mlir::BlockArgument)> m_undefinedCallback;
    std::function<mlir::Value(mlir::Value, mlir::Value)> m_blockArgMergeOptCallback;

    mlir::Value tryRemoveTrivialBlockArgument(mlir::BlockArgument argument);

    mlir::Value addBlockArguments(DefinitionsMap& map, mlir::BlockArgument argument);

    mlir::Value readVariableRecursive(mlir::Location loc, mlir::Type type, DefinitionsMap& map, mlir::Block* block);

    void removeBlockArgumentOperands(mlir::BlockArgument argument);

public:
    explicit SSABuilder(
        std::function<mlir::Value(mlir::BlockArgument)> undefinedCallback = [](auto) -> mlir::Value
        { PYLIR_UNREACHABLE; },
        std::function<mlir::Value(mlir::Value, mlir::Value)> blockArgMergeOptCallback = {})
        : m_undefinedCallback(std::move(undefinedCallback)),
          m_blockArgMergeOptCallback(std::move(blockArgMergeOptCallback))
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

    bool isOpenBlock(mlir::Block* block) const
    {
        return m_openBlocks.count(block);
    }

    void markOpenBlock(mlir::Block* block);

    void sealBlock(mlir::Block* block);

    mlir::Value readVariable(mlir::Location loc, mlir::Type type, DefinitionsMap& map, mlir::Block* block);
};
} // namespace pylir
