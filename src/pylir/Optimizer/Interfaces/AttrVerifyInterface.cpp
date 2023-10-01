// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AttrVerifyInterface.hpp"

#include "pylir/Optimizer/Interfaces/AttrVerifyInterface.cpp.inc"

using namespace mlir;
using namespace pylir;

LogicalResult AttrVerifyInterface::verify(Operation* operation,
                                          SymbolTableCollection& table) {
  AttrTypeWalker walker;

  // Current sub-operation set by the operation walker from which the attribute
  // is reachable.
  Operation* currentOperation;

  walker.addWalk([&](AttrVerifyInterface verifyInterface) {
    if (failed(verifyInterface.verifyStructure(currentOperation, table)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  // Use a post-order walk to first verify any sub-attributes and sub-operations
  // prior to verifying the attribute and operation itself.
  constexpr WalkOrder walkOrder = WalkOrder::PostOrder;

  // Walk all operations including itself. This includes both inherent and
  // discard-able attributes. The walk result from the attribute type walker
  // is forwarded to the operation walker.
  WalkResult result = operation->walk<walkOrder>([&](Operation* subOperation) {
    // Set the current operation for the `AttrTypeWalker` instance above.
    currentOperation = subOperation;
    return walker.walk<walkOrder>(currentOperation->getAttrDictionary());
  });

  // The walking was interrupted if any verification failed. Propagate the
  // failure.
  return failure(/*isFailure=*/result.wasInterrupted());
}
