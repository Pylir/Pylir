//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "InlineCost.hpp"

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Visitors.h>

pylir::InlineCost::InlineCost(mlir::MLIRContext* context)
    : m_collection(context) {}

std::size_t pylir::InlineCost::getCostOf(mlir::Operation* operation,
                                         bool recurse) {
  std::size_t size = 0;

  auto sizeOfImpl = [&](mlir::Operation* op) {
    if (op->hasTrait<mlir::OpTrait::ConstantLike>())
      return;

    const DialectInlineCostInterface* interface =
        m_collection.getInterfaceFor(op);
    if (!interface) {
      size += 5;
      return;
    }
    size += interface->getCost(op);
  };

  sizeOfImpl(operation);
  if (recurse)
    operation->walk(sizeOfImpl);

  return size;
}

pylir::InlineCostAnalysis::InlineCostAnalysis(mlir::Operation* operation) {
  m_cost = InlineCost(operation->getContext())
               .getCostOf(operation, /*recurse=*/true);
}
