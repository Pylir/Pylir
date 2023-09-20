//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ExternalModels.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/Matchers.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/Interfaces/ConditionalBranchInterface.hpp>
#include <pylir/Optimizer/Interfaces/DialectImplicationPatternsInterface.hpp>
#include <pylir/Optimizer/Interfaces/DialectInlineCostInterface.hpp>

namespace {

struct CondBranchOpConditionalBranchInterface
    : public pylir::ConditionalBranchInterface::ExternalModel<
          CondBranchOpConditionalBranchInterface, mlir::cf::CondBranchOp> {
  llvm::SmallVector<std::pair<mlir::Block*, mlir::Attribute>>
  getBranchImplications(mlir::Operation* op) const {
    auto branchOp = mlir::cast<mlir::cf::CondBranchOp>(op);
    return {
        {branchOp.getTrueDest(), mlir::BoolAttr::get(op->getContext(), true)},
        {branchOp.getFalseDest(), mlir::BoolAttr::get(op->getContext(), false)},
    };
  }

  mlir::Value getCondition(mlir::Operation* op) const {
    return mlir::cast<mlir::cf::CondBranchOp>(op).getCondition();
  }
};

struct ControlFlowInlineCostInterface
    : public pylir::DialectInlineCostInterface {
  using pylir::DialectInlineCostInterface::DialectInlineCostInterface;

  std::size_t getCost(mlir::Operation* op) const override {
    return llvm::TypeSwitch<mlir::Operation*, std::size_t>(op)
        .Case<mlir::cf::BranchOp, mlir::cf::AssertOp>([](auto) { return 0; })
        .Default(std::size_t{5});
  }
};

struct ArithImplicationPatterns
    : public pylir::DialectImplicationPatternsInterface {
  using pylir::DialectImplicationPatternsInterface::
      DialectImplicationPatternsInterface;

  void getImplicationPatterns(
      pylir::PatternAllocator&, mlir::Value conditional, mlir::Attribute value,
      llvm::function_ref<void(pylir::ImplicationPatternBase*)>,
      llvm::function_ref<void(mlir::Value, mlir::Attribute)>
          implicationAddCallback) const override {
    auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(value);
    if (!boolAttr)
      return;

    mlir::Value notOperand;
    if (mlir::matchPattern(conditional, mlir::m_Op<mlir::arith::XOrIOp>(
                                            mlir::matchers::m_Any(&notOperand),
                                            mlir::m_One()))) {
      implicationAddCallback(
          notOperand, mlir::BoolAttr::get(getContext(), !boolAttr.getValue()));
      return;
    }

    if (boolAttr.getValue()) {
      if (auto andOp = conditional.getDefiningOp<mlir::arith::AndIOp>()) {
        implicationAddCallback(andOp.getLhs(), boolAttr);
        implicationAddCallback(andOp.getRhs(), boolAttr);
      }
    } else {
      if (auto orOp = conditional.getDefiningOp<mlir::arith::OrIOp>()) {
        implicationAddCallback(orOp.getLhs(), boolAttr);
        implicationAddCallback(orOp.getRhs(), boolAttr);
      }
    }
  }
};

} // namespace

void pylir::registerExternalModels(mlir::DialectRegistry& registry) {
  registry.addExtension(
      +[](mlir::MLIRContext*, mlir::cf::ControlFlowDialect* dialect) {
        dialect->addInterface<ControlFlowInlineCostInterface>();
        mlir::cf::CondBranchOp::attachInterface<
            CondBranchOpConditionalBranchInterface>(*dialect->getContext());
      });
  registry.addExtension(
      +[](mlir::MLIRContext*, mlir::arith::ArithDialect* dialect) {
        dialect->addInterface<ArithImplicationPatterns>();
      });
}
