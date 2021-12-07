#pragma once

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>

#define GET_TYPEDEF_CLASSES
#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsTypes.h.inc"

namespace pylir::Py
{
inline mlir::FunctionType getUniversalCCType(mlir::MLIRContext* context)
{
    auto dynamicType = Py::DynamicType::get(context);
    return mlir::FunctionType::get(context, mlir::TypeRange{dynamicType, dynamicType, dynamicType}, {dynamicType});
}
} // namespace pylir::Py
