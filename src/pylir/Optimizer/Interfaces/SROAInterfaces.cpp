//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "SROAInterfaces.hpp"

#include <mlir/IR/Matchers.h>

#include "pylir/Optimizer/Interfaces/SROAAttrInterfaces.cpp.inc"
#include "pylir/Optimizer/Interfaces/SROAOpInterfaces.cpp.inc"

mlir::LogicalResult
pylir::aggregateUseCanParticipateInSROA(const mlir::OpOperand& aggregateUse) {
  auto op = mlir::dyn_cast_or_null<pylir::SROAReadWriteOpInterface>(
      aggregateUse.getOwner());
  if (!op || &op.getAggregateOperand() != &aggregateUse)
    return mlir::failure();

  return op.getSROAKey();
}
