// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_INTERFACES_CONDITIONAL_BRANCH_INTERFACE
#define PYLIR_INTERFACES_CONDITIONAL_BRANCH_INTERFACE

include "mlir/IR/OpBase.td"

def ConditionalBranchInterface : OpInterface<"ConditionalBranchInterface"> {
  let cppNamespace = "::pylir";

  let description = [{
    Interface that can be used by passes to get implied values from a branching
    terminator instructions.
    Example for this includes eg. a conditional branch instruction.
    It implies that the value for its condition is true, within the path of its
    true destination and false within the path of its false destination, until
    some merge point has been reached.
  }];

  let methods = [
    InterfaceMethod<[{
      Method called to get the implications of the branch instruction.
      It should simply return a list of pairs consisting of a successor and the
      constant the value returned by `getCondition` has for the given successor.
      Implementations itself should not care about contradicting or duplicate
      successors.
    }], "::llvm::SmallVector<std::pair<mlir::Block*, mlir::Attribute>>", "getBranchImplications", (ins)>,

    InterfaceMethod<[{
      Returns the condition value that is used to decide the branch and whose
      value is implied within the operations successors.
    }], "mlir::Value", "getCondition", (ins)>,
  ];

  let verify = [{
    static_assert(ConcreteOp::template hasTrait<mlir::OpTrait::IsTerminator>());
    return mlir::success();
  }];
}


#endif
