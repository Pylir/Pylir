#pragma once

#include <mlir/IR/DialectInterface.h>

namespace pylir
{
class DialectCostInterface : public mlir::DialectInterface::Base<DialectCostInterface>
{
public:
    DialectCostInterface(mlir::Dialect* dialect) : Base(dialect) {}

    virtual std::size_t getCost(mlir::Operation* op) const = 0;
};
} // namespace pylir
