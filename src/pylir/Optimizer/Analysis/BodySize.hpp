
#pragma once

#include <mlir/IR/Operation.h>

namespace pylir
{
class BodySize
{
    std::size_t m_size;

public:
    explicit BodySize(mlir::Operation* operation);

    std::size_t getSize() const
    {
        return m_size;
    }
};
} // namespace pylir
