#include "BodySize.hpp"

pylir::BodySize::BodySize(mlir::Operation* operation) : m_size(0)
{
    operation->walk([&](mlir::Operation*) { m_size++; });
}
