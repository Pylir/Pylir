//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "SSABuilder.hpp"

#include <mlir/Interfaces/ControlFlowInterfaces.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Support/Macros.hpp>

void pylir::SSABuilder::markOpenBlock(mlir::Block* block) {
  m_openBlocks.insert({block, {}});
}

void pylir::SSABuilder::sealBlock(mlir::Block* block) {
  auto result = m_openBlocks.find(block);
  if (result == m_openBlocks.end())
    return;

  auto blockArgs =
      llvm::to_vector(block->getArguments().take_back(result->second.size()));
  for (auto [blockArgument, map] : llvm::zip(blockArgs, result->second))
    addBlockArguments(*map, blockArgument);

  m_openBlocks.erase(result);
}

mlir::Value pylir::SSABuilder::addBlockArguments(InternalDefinitionsMap& map,
                                                 mlir::BlockArgument argument) {
  for (auto pred = argument.getOwner()->pred_begin();
       pred != argument.getOwner()->pred_end(); pred++) {
    auto terminator =
        mlir::cast<mlir::BranchOpInterface>((*pred)->getTerminator());
    terminator.getSuccessorOperands(pred.getSuccessorIndex())
        .append(
            readVariable(argument.getLoc(), argument.getType(), map, *pred));
  }
  return tryRemoveTrivialBlockArgument(argument);
}

void pylir::SSABuilder::removeBlockArgumentOperands(
    mlir::BlockArgument argument) {
  for (auto pred = argument.getOwner()->pred_begin();
       pred != argument.getOwner()->pred_end(); pred++) {
    auto terminator =
        mlir::cast<mlir::BranchOpInterface>((*pred)->getTerminator());
    terminator.getSuccessorOperands(pred.getSuccessorIndex())
        .erase(argument.getArgNumber());
  }
}

mlir::Value
pylir::SSABuilder::tryRemoveTrivialBlockArgument(mlir::BlockArgument argument) {
  llvm::SmallVector<mlir::Value> operands;
  for (auto pred = argument.getOwner()->pred_begin();
       pred != argument.getOwner()->pred_end(); pred++) {
    auto terminator =
        mlir::cast<mlir::BranchOpInterface>((*pred)->getTerminator());
    auto ops = terminator.getSuccessorOperands(pred.getSuccessorIndex());
    operands.push_back(ops[argument.getArgNumber()]);
  }
  mlir::Value same =
      optimizeBlockArgsOperands(operands, argument, argument.getOwner(),
                                argument.getType(), argument.getLoc());
  if (!same)
    return argument;

  removeBlockArgumentOperands(argument);
  same = replaceBlockArgument(argument, same);

  return same;
}

mlir::Value
pylir::SSABuilder::replaceBlockArgument(mlir::BlockArgument argument,
                                        mlir::Value replacement) {
  llvm::SmallVector<ValueTracker> dependentBlockArgs;
  for (auto& user : argument.getUses()) {
    auto branch = mlir::dyn_cast<mlir::BranchOpInterface>(user.getOwner());
    if (!branch)
      continue;

    // Avoid duplicates in 'dependentBlockArgs'. We could use a SetVector in the
    // future as well, but we expect the array to be small generally speaking.
    if (auto ops = branch.getSuccessorBlockArgument(user.getOperandNumber());
        ops && !llvm::is_contained(dependentBlockArgs, *ops))
      dependentBlockArgs.emplace_back(*ops);
  }

  argument.replaceAllUsesWith(replacement);
  argument.getOwner()->eraseArgument(argument.getArgNumber());

  for (mlir::Value ba : dependentBlockArgs) {
    // If the block argument was replaced (in previous iterations), it might not
    // be a 'BlockArgument' anymore. If it was replaced with another
    // 'BlockArgument' we are doing more necessary work here, but it does not
    // impact optimality nor correctness.
    auto arg = mlir::dyn_cast<mlir::BlockArgument>(ba);
    if (!arg)
      continue;

    if (arg == replacement)
      replacement = tryRemoveTrivialBlockArgument(arg);
    else
      tryRemoveTrivialBlockArgument(arg);
  }
  return replacement;
}

mlir::Value pylir::SSABuilder::optimizeBlockArgsOperands(
    llvm::ArrayRef<mlir::Value> operands, mlir::BlockArgument maybeArgument,
    mlir::Block* block, mlir::Type type, mlir::Location loc) {
  mlir::Value same;
  for (auto blockOperand : operands) {
    if (blockOperand == same || blockOperand == maybeArgument)
      continue;

    if (same) {
      if (!m_blockArgMergeOptCallback)
        return nullptr;

      if (auto merge = m_blockArgMergeOptCallback(same, blockOperand)) {
        same = merge;
        continue;
      }
      return nullptr;
    }
    same = blockOperand;
  }
  if (!same)
    return m_undefinedCallback(block, type, loc);

  return same;
}

mlir::Value pylir::SSABuilder::readVariable(mlir::Location loc, mlir::Type type,
                                            InternalDefinitionsMap& map,
                                            mlir::Block* block) {
  if (auto result = map.find(block); result != map.end())
    return result->second;

  if (auto result = m_marked.find(block); result != m_marked.end())
    return map[block] = result->second = block->addArgument(type, loc);

  return readVariableRecursive(loc, type, map, block);
}

mlir::Value
pylir::SSABuilder::readVariableRecursive(mlir::Location loc, mlir::Type type,
                                         InternalDefinitionsMap& map,
                                         mlir::Block* block) {
  if (auto result = m_openBlocks.find(block); result != m_openBlocks.end()) {
    mlir::Value val = block->addArgument(type, loc);
    result->second.emplace_back(&map);
    return map[block] = val;
  }

  // Single predecessor is trivial as no block arguments have to be created.
  if (auto* uniquePredecessor = block->getUniquePredecessor())
    return map[block] = readVariable(loc, type, map, uniquePredecessor);

  // Mark the block to catch loops in 'readVariable' below. If marked and
  // required, it'll create a BlockArgument.
  m_marked[block] = nullptr;
  llvm::SmallVector<mlir::Value> predArgs;
  llvm::transform(
      block->getPredecessors(), std::back_inserter(predArgs),
      [&](mlir::Block* pred) { return readVariable(loc, type, map, pred); });

  // Note: Have to use 'find' again and couldn't have done so above and saved a
  // lookup, because DenseMap invalidates iterators and references on insert,
  // which may have occurred in 'readVariable'.
  auto iter = m_marked.find(block);
  mlir::BlockArgument maybeArgument = iter->second;
  m_marked.erase(iter);

  if (auto val = optimizeBlockArgsOperands(predArgs, maybeArgument, block, type,
                                           loc)) {
    if (maybeArgument) {
      // If we were in a loop and still managed to optimize it we have to
      // replace the block argument with the single unique value. This may also
      // trigger the possibility of optimizing more block arguments that used
      // this block arg as operand.
      val = replaceBlockArgument(maybeArgument, val);
    }
    return map[block] = val;
  }

  // Create the block arg if that hasn't yet happened due to a loop.
  if (!maybeArgument)
    maybeArgument = block->addArgument(type, loc);

  std::size_t counter = 0;
  for (auto pred = block->pred_begin(); pred != block->pred_end(); pred++) {
    auto terminator =
        mlir::cast<mlir::BranchOpInterface>((*pred)->getTerminator());
    terminator.getSuccessorOperands(pred.getSuccessorIndex())
        .append(predArgs[counter++]);
  }
  return map[block] = maybeArgument;
}
