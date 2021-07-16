#pragma once

#include <mlir/IR/Types.h>

#include <vector>

namespace pylir::Dialect::detail
{
std::vector<mlir::Type> variantUnion(llvm::ArrayRef<mlir::Type> types);
}

#define GET_TYPEDEF_CLASSES
#include "pylir/Optimizer/Dialect/PylirOpsTypes.h.inc"

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
