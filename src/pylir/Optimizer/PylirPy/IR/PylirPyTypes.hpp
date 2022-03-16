#pragma once

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>

#include "ObjectTypeInterface.hpp"

#define GET_TYPEDEF_CLASSES
#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsTypes.h.inc"

namespace pylir::Py
{
inline mlir::FunctionType getUniversalCCType(mlir::MLIRContext* context)
{
    auto unknownType = Py::UnknownType::get(context);
    return mlir::FunctionType::get(context, mlir::TypeRange{unknownType, unknownType, unknownType}, {unknownType});
}

pylir::Py::ObjectTypeInterface joinTypes(pylir::Py::ObjectTypeInterface lhs, pylir::Py::ObjectTypeInterface rhs);
} // namespace pylir::Py
