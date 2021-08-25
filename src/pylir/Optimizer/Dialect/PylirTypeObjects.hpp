
#pragma once

#include "PylirOps.hpp"

namespace pylir::Dialect
{
constexpr std::string_view objectTypeObjectName = "__builtins__.object";

Dialect::ConstantGlobalOp getObjectTypeObject(mlir::ModuleOp& module);

constexpr std::string_view typeTypeObjectName = "__builtins__.type";

Dialect::ConstantGlobalOp getTypeTypeObject(mlir::ModuleOp& module);

constexpr std::string_view functionTypeObjectName = "__builtins__.function";

Dialect::ConstantGlobalOp getFunctionTypeObject(mlir::ModuleOp& module);

constexpr std::string_view intTypeObjectName = "__builtins__.int";

Dialect::ConstantGlobalOp getIntTypeObject(mlir::ModuleOp& module);

constexpr std::string_view noneTypeObjectName = "__builtins__.NoneType";

Dialect::ConstantGlobalOp getNoneTypeObject(mlir::ModuleOp& module);

Dialect::ConstantGlobalOp getNoneObject(mlir::ModuleOp& module);

constexpr std::string_view notImplementedTypeObjectName = "__builtins__.NotImplementedType";

Dialect::ConstantGlobalOp getNotImplementedTypeObject(mlir::ModuleOp& module);

Dialect::ConstantGlobalOp getNotImplementedObject(mlir::ModuleOp& module);

constexpr std::string_view tupleTypeObjectName = "__builtins__.tuple";

Dialect::ConstantGlobalOp getTupleTypeObject(mlir::ModuleOp& module);

constexpr std::string_view stringTypeObjectName = "__builtins__.str";

Dialect::ConstantGlobalOp getStringTypeObject(mlir::ModuleOp& module);

constexpr std::string_view boolTypeObjectName = "__builtins__.bool";

Dialect::ConstantGlobalOp getBoolTypeObject(mlir::ModuleOp& module);

mlir::FunctionType getCCFuncType(mlir::MLIRContext* context);
} // namespace pylir::Dialect
