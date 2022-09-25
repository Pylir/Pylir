//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "BodySize.hpp"

#include <mlir/IR/OpDefinition.h>

#include <pylir/Optimizer/Interfaces/DialectCostInterface.hpp>

pylir::BodySize::BodySize(mlir::Operation* operation)
{
    mlir::DialectInterfaceCollection<pylir::DialectCostInterface> collection(operation->getContext());
    operation->walk(
        [&](mlir::Operation* op)
        {
            if (op->hasTrait<mlir::OpTrait::ConstantLike>())
            {
                return;
            }
            const auto* interface = collection.getInterfaceFor(op);
            if (!interface)
            {
                m_size++;
                return;
            }
            m_size += interface->getCost(op);
        });
}
