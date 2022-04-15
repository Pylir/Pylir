#include "BodySize.hpp"

#include <mlir/IR/OpDefinition.h>

#include <pylir/Optimizer/Interfaces/DialectCostInterface.hpp>

pylir::BodySize::BodySize(mlir::Operation* operation) : m_size(0)
{
    mlir::DialectInterfaceCollection<pylir::DialectCostInterface> collection(operation->getContext());
    operation->walk(
        [&](mlir::Operation* op)
        {
            if (op->hasTrait<mlir::OpTrait::ConstantLike>())
            {
                return;
            }
            auto* interface = collection.getInterfaceFor(op);
            if (!interface)
            {
                m_size++;
                return;
            }
            m_size += interface->getCost(op);
        });
}
