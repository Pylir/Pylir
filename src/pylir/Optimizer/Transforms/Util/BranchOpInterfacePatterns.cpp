//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "BranchOpInterfacePatterns.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>

#include <pylir/Support/Macros.hpp>

namespace {

void replaceAllUsesWith(mlir::PatternRewriter& rewriter, mlir::Block* toReplace,
                        mlir::Block* replacement) {
  for (auto& iter : llvm::make_early_inc_range(toReplace->getUses()))
    rewriter.modifyOpInPlace(iter.getOwner(), [&] { iter.set(replacement); });
}

// bb0:
// br ^bb1()
//
// bb1:
// br ^bb2(%args...)
//
// ->
// bb0:
// br ^bb2(%args...)
struct PassthroughArgRemove
    : mlir::OpInterfaceRewritePattern<mlir::BranchOpInterface> {
  using mlir::OpInterfaceRewritePattern<
      mlir::BranchOpInterface>::OpInterfaceRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::BranchOpInterface op,
                  mlir::PatternRewriter& rewriter) const override {
    bool changed = false;
    for (const auto& iter : llvm::enumerate(op->getSuccessors())) {
      if (iter.value()->getNumArguments() != 0)
        continue;

      auto brOp = mlir::dyn_cast_or_null<mlir::BranchOpInterface>(
          iter.value()->getTerminator());
      auto memoryEffect = mlir::dyn_cast_or_null<mlir::MemoryEffectOpInterface>(
          iter.value()->getTerminator());
      // If the branch op has just a single successor, no side effects, and no
      // operands, it can only be a simple branch op.
      if (!brOp || !memoryEffect || brOp->getNumSuccessors() != 1 ||
          !memoryEffect.hasNoEffect() || brOp->getNumOperands() != 0)
        continue;

      // If the block has more than one operation (aka more than just the
      // terminator), the optimization is not valid as the produced values may
      // not be calculated nor their side effects executed.
      if (!llvm::hasSingleElement(iter.value()->getOperations()))
        continue;

      auto successorOperands = brOp.getSuccessorOperands(0);
      if (llvm::any_of(successorOperands.getForwardedOperands(),
                       [op](mlir::Value value) {
                         return value.getDefiningOp() == op;
                       }) ||
          successorOperands.getProducedOperandCount() != 0)
        continue;
      changed = true;
      rewriter.modifyOpInPlace(op, [&] {
        op->setSuccessor(brOp->getSuccessor(0), iter.index());
        op.getSuccessorOperands(iter.index())
            .append(successorOperands.getForwardedOperands());
      });
    }
    return mlir::success(changed);
  }
};

// bb0:
// br ^bb1(%0) as well as other predecessors using %0
//
// bb1(%1):
// br ^bb1(%0) or br ^bb1(%1)
//
// ->
// bb0:
// br ^bb1()
//
// bb1:
// br bb1()
struct TrivialBlockArgRemove
    : mlir::OpInterfaceRewritePattern<mlir::BranchOpInterface> {
  using mlir::OpInterfaceRewritePattern<
      mlir::BranchOpInterface>::OpInterfaceRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::BranchOpInterface op,
                  mlir::PatternRewriter& rewriter) const override {
    bool changed = false;
    for (const auto& [index, succ] : llvm::enumerate(op->getSuccessors())) {
      llvm::SmallSetVector<mlir::BlockArgument, 4> skipList;

      mlir::SuccessorOperands successorOperands =
          op.getSuccessorOperands(index);
      for (std::size_t i = successorOperands.getProducedOperandCount();
           i < successorOperands.size(); i++) {
        auto succBlockArg =
            *op.getSuccessorBlockArgument(successorOperands.getOperandIndex(i));
        mlir::Value same;
        for (auto pred = succBlockArg.getOwner()->pred_begin();
             pred != succBlockArg.getOwner()->pred_end(); pred++) {
          auto terminator =
              mlir::dyn_cast<mlir::BranchOpInterface>((*pred)->getTerminator());
          if (!terminator) {
            same = nullptr;
            break;
          }
          auto ops = terminator.getSuccessorOperands(pred.getSuccessorIndex());
          auto blockOperand = ops[succBlockArg.getArgNumber()];
          if (!blockOperand) {
            same = nullptr;
            break;
          }
          if (blockOperand == same || blockOperand == succBlockArg)
            continue;

          if (same) {
            same = nullptr;
            break;
          }
          same = blockOperand;
        }
        if (!same)
          continue;

        rewriter.replaceAllUsesWith(succBlockArg, same);
        skipList.insert(succBlockArg);
      }
      if (skipList.empty())
        continue;

      changed = true;
      auto* newSucc = rewriter.splitBlock(succ, succ->begin());
      auto remainingBlockArgs = llvm::make_filter_range(
          succ->getArguments(),
          [&](mlir::BlockArgument arg) { return !skipList.contains(arg); });

      newSucc->addArguments(
          llvm::to_vector(llvm::map_range(
              remainingBlockArgs, std::mem_fn(&mlir::BlockArgument::getType))),
          llvm::to_vector(llvm::map_range(
              remainingBlockArgs, std::mem_fn(&mlir::BlockArgument::getLoc))));

      for (auto [old, newBlockArg] : llvm::zip(
               llvm::to_vector(remainingBlockArgs), newSucc->getArguments()))
        rewriter.replaceAllUsesWith(old, newBlockArg);

      for (auto pred = succ->pred_begin(); pred != succ->pred_end(); pred++) {
        auto terminator =
            mlir::cast<mlir::BranchOpInterface>((*pred)->getTerminator());
        auto operands =
            terminator.getSuccessorOperands(pred.getSuccessorIndex());

        rewriter.modifyOpInPlace(terminator, [&] {
          for (mlir::BlockArgument iter : llvm::reverse(skipList))
            operands.erase(iter.getArgNumber());
        });
      }
      replaceAllUsesWith(rewriter, succ, newSucc);
      rewriter.eraseBlock(succ);
    }
    return mlir::success(changed);
  }
};

} // namespace

void pylir::populateWithBranchOpInterfacePattern(
    mlir::RewritePatternSet& patterns) {
  patterns.insert<PassthroughArgRemove, TrivialBlockArgRemove>(
      patterns.getContext());
}
