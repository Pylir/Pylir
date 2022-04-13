#include "BodySize.hpp"

#include <mlir/IR/OpDefinition.h>

pylir::BodySize::BodySize(mlir::Operation* operation) : m_size(0)
{
    operation->walk(
        [&](mlir::Operation* op)
        {
            if (op->hasTrait<mlir::OpTrait::ConstantLike>())
            {
                return;
            }
            m_size++;
        });
}
