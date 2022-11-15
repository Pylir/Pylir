//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ControlFlowInlineCostModel.hpp"

#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/Interfaces/DialectInlineCostInterface.hpp>

namespace
{
struct ControlFlowInlineCostInterface : public pylir::DialectInlineCostInterface
{
    using pylir::DialectInlineCostInterface::DialectInlineCostInterface;

    std::size_t getCost(mlir::Operation* op) const override
    {
        return llvm::TypeSwitch<mlir::Operation*, std::size_t>(op)
            .Case<mlir::cf::BranchOp, mlir::cf::AssertOp>([](auto) { return 0; })
            .Default(std::size_t{5});
    }
};
} // namespace

void pylir::registerControlFlowInlineCostModel(mlir::DialectRegistry& registry)
{
    registry.addExtension(+[](mlir::MLIRContext*, mlir::cf::ControlFlowDialect* dialect)
                          { dialect->addInterface<ControlFlowInlineCostInterface>(); });
}
