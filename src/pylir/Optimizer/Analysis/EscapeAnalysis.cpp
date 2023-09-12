// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "EscapeAnalysis.hpp"

#include <mlir/Interfaces/ControlFlowInterfaces.h>

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>

#include <pylir/Optimizer/Interfaces/CaptureInterface.hpp>

#include <cstddef>

bool pylir::EscapeAnalysis::escapes(mlir::Value value) {
  constexpr auto rootValueIndex = static_cast<std::size_t>(-1);

  llvm::SmallVector<std::pair<mlir::Value, std::size_t>> dfsStack;
  dfsStack.emplace_back(value, rootValueIndex);

  auto markParentChainEscaping = [&](std::size_t currIndex) {
    while (true) {
      m_results[dfsStack[currIndex].first] = true;
      if (dfsStack[currIndex].second == rootValueIndex)
        return;

      currIndex = dfsStack[currIndex].second;
    }
  };

  llvm::SmallPtrSet<mlir::Value, 8> seen;
  while (!dfsStack.empty()) {
    mlir::Value currValue = dfsStack.back().first;
    std::size_t currIndex = dfsStack.size() - 1;

    auto iter = m_results.find(currValue);
    if (iter != m_results.end()) {
      if (!iter->second) {
        dfsStack.pop_back();
        continue;
      }
      markParentChainEscaping(currIndex);
      return true;
    }
    if (!seen.insert(currValue).second) {
      // If this is the second time we see this item on the stack it means that
      // all the block arguments it is passed to do not escape either and have
      // therefore been pushed off the stack. This value therefore does not
      // escape either.
      m_results[currValue] = false;
      dfsStack.pop_back();
      seen.erase(currValue);
      continue;
    }

    for (mlir::OpOperand& use : currValue.getUses()) {
      mlir::Operation* user = use.getOwner();
      if (auto branch = mlir::dyn_cast<mlir::BranchOpInterface>(user)) {
        if (auto blockArg =
                branch.getSuccessorBlockArgument(use.getOperandNumber())) {
          // We mustn't add the block arg if it has already been seen as it is
          // already on the stack. Adding it a second time would lead to the
          // loop falsely concluding it does not escape, despite likely not
          // having been fully processed yet.
          if (!seen.contains(*blockArg))
            dfsStack.emplace_back(*blockArg, currIndex);

          continue;
        }
      }

      // For the time being we make the conservative and easy choice of saying
      // any captured value also escapes. This mustn't be true however, and
      // could be further analyzed in the future, although possibly difficult
      // and/or expensive in the general case.
      // E.g. if a value is captured by a dictionary does not imply it must
      // escape. It then depends on how this dictionary is used. If the
      // dictionary escapes it implies that the contained values also escape. If
      // the dictionary does not escape, the values within may still escape.
      auto capture = mlir::dyn_cast<pylir::CaptureInterface>(user);
      if (!capture || capture.capturesValue(currValue)) {
        markParentChainEscaping(currIndex);
        return true;
      }
    }
    // If we reached the end of the loop it means the value either does not
    // escape or is dependent on the analysis of block arguments. Either way we
    // can simply fallthrough here and the "seen" case will take care of
    // calculating the correct result either way.
  }
  return false;
}
