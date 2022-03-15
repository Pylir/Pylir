#pragma once

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>

#include "ObjectTypeInterface.hpp"

namespace pylir::Py
{
template <class ConcreteType>
class ObjectType : public mlir::TypeTrait::TraitBase<ConcreteType, ObjectType>
{
};
} // namespace pylir::Py

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
