//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "InlinerUtil.hpp"

#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/IRMapping.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>

namespace {
void remapInlinedLocations(mlir::Region& inlinedRegion,
                           mlir::Location callerLoc) {
  llvm::DenseMap<mlir::Location, mlir::Location> mappedLocations;
  auto remapOpLoc = [&](mlir::Operation* op) {
    auto [iter, inserted] =
        mappedLocations.try_emplace(op->getLoc(), op->getLoc());
    if (inserted)
      iter->second = mlir::CallSiteLoc::get(op->getLoc(), callerLoc);

    op->setLoc(iter->second);
  };
  for (mlir::Block& block : inlinedRegion)
    block.walk(remapOpLoc);
}
} // namespace

mlir::IRMapping pylir::Py::inlineCall(mlir::CallOpInterface call,
                                      mlir::CallableOpInterface callable) {
  auto exceptionHandler =
      mlir::dyn_cast<pylir::Py::ExceptionHandlingInterface>(*call);

  mlir::IRMapping mapping;
  mlir::Region* callableRegion = callable.getCallableRegion();
  mapping.map(callableRegion->getArguments(), call.getArgOperands());

  // We first copy the callable into a temporary region. This is because the
  // callable region might actually be equal to the region of the caller! This
  // is the case when doing an inlining operation on a direct recursive
  // function. This way we minimize the modifications visible in the caller
  // region, while transforming the inlined ops.
  mlir::Region tempClone;
  callableRegion->cloneInto(&tempClone, mapping);

  // Split the caller block as we require the successor block after the call to
  // replace return instructions.
  mlir::Block* preBlock = call->getBlock();
  mlir::Block* postBlock = preBlock->splitBlock(call);
  postBlock->addArguments(
      callable.getResultTypes(),
      llvm::to_vector(llvm::map_range(
          call->getResults(), [](mlir::Value arg) { return arg.getLoc(); })));

  // Builder constructed here as we cannot get the context from any blocks
  // within 'tempClone'.
  mlir::OpBuilder builder(call->getContext());

  // We iterate over the callable region (aka the source of the clone) for the
  // simple reason that we want to update the mapper as we are transforming the
  // inlined blocks. At this point in time the inlined ops are still
  // structurally equal to the source ops.
  for (auto& sourceBlock : *callableRegion) {
    if (exceptionHandler) {
      for (auto sourceOp :
           sourceBlock.getOps<Py::AddableExceptionHandlingInterface>()) {
        auto op = mlir::cast<Py::AddableExceptionHandlingInterface>(
            mapping.lookup(sourceOp.getOperation()));
        mlir::Block* block = op->getBlock();

        mlir::Block* successBlock = block->splitBlock(op->getIterator());
        builder.setInsertionPointToEnd(block);
        mlir::Operation* newOp = op.cloneWithExceptionHandling(
            builder, successBlock, exceptionHandler.getExceptionPath(),
            exceptionHandler.getUnwindDestOperands());
        op->replaceAllUsesWith(newOp);
        mapping.map(sourceOp.getOperation(), newOp);
        mapping.map(sourceOp->getResults(), newOp->getResults());
        mlir::Location loc = op->getLoc();
        op.erase();
        // The success block may be empty if 'op' was a terminator without
        // successors. Create an unreachable op in this case to terminate it.
        if (successBlock->empty()) {
          mlir::OpBuilder::InsertionGuard guard{builder};
          builder.setInsertionPointToEnd(successBlock);
          builder.create<UnreachableOp>(loc);
        }
      }
    }
    // While we are iterating over the callable region, it may actually contain
    // empty blocks or blocks without a terminator as in a direct recursive
    // function call, the callable region is equal to the caller region. Since
    // we did a block split in the caller region, we may have modified the
    // callable region as well and left it in a currently invalid state.
    if (sourceBlock.empty())
      continue;

    mlir::Operation* sourceTerminator = &sourceBlock.back();
    if (sourceTerminator->hasTrait<mlir::OpTrait::ReturnLike>()) {
      mlir::Operation* terminator = mapping.lookup(sourceTerminator);

      builder.setInsertionPoint(terminator);
      auto branch = builder.create<mlir::cf::BranchOp>(
          terminator->getLoc(), postBlock, terminator->getOperands());
      mapping.map(sourceTerminator, branch.getOperation());
      terminator->erase();
      continue;
    }
  }

  remapInlinedLocations(tempClone, call->getLoc());
  // Finally move all the blocks from 'tempClone' into the caller.
  call->getParentRegion()->getBlocks().splice(postBlock->getIterator(),
                                              tempClone.getBlocks());

  mlir::Block* firstInlinedBlock = preBlock->getNextNode();

  // Move the operations from the first inlined block into the block of the
  // call-site. Since the first inlined block corresponds to the entry block,
  // call-site block is the only successor and there is no need to create a
  // branch operation here.
  preBlock->getOperations().splice(preBlock->end(),
                                   firstInlinedBlock->getOperations());
  // Erase the firstInlinedBlock from the mapping as well since we are about to
  // erase it.
  mapping.erase(&callableRegion->front());
  firstInlinedBlock->erase();
  for (auto [res, arg] :
       llvm::zip(call->getResults(), postBlock->getArguments()))
    res.replaceAllUsesWith(arg);

  if (exceptionHandler) {
    builder.setInsertionPoint(call);
    builder.create<mlir::cf::BranchOp>(
        call->getLoc(), exceptionHandler.getHappyPath(),
        exceptionHandler.getNormalDestOperands());
  }
  mapping.erase(call);
  call.erase();
  return mapping;
}
