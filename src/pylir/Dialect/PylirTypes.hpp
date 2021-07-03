#pragma once

#include <mlir/IR/Types.h>

#define GET_TYPEDEF_CLASSES
#include "pylir/Dialect/PylirOpsTypes.h.inc"

namespace pylir::Dialect
{
inline bool isNumbers(mlir::Type type)
{
    return type.isa<IntegerType, BoolType, FloatType>();
}

inline bool isIntegerLike(mlir::Type type)
{
    return type.isa<IntegerType, BoolType>();
}

} // namespace pylir::Dialect
