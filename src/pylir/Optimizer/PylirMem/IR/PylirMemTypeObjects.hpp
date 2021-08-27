
#pragma once

#include "PylirMemOps.hpp"

namespace pylir::Mem
{
constexpr std::string_view objectTypeObjectName = "__builtins__.object";

Mem::ConstantGlobalOp getObjectTypeObject(mlir::ModuleOp& module);

constexpr std::string_view typeTypeObjectName = "__builtins__.type";

Mem::ConstantGlobalOp getTypeTypeObject(mlir::ModuleOp& module);

constexpr std::string_view functionTypeObjectName = "__builtins__.function";

Mem::ConstantGlobalOp getFunctionTypeObject(mlir::ModuleOp& module);

constexpr std::string_view intTypeObjectName = "__builtins__.int";

Mem::ConstantGlobalOp getIntTypeObject(mlir::ModuleOp& module);

constexpr std::string_view noneTypeObjectName = "__builtins__.NoneType";

Mem::ConstantGlobalOp getNoneTypeObject(mlir::ModuleOp& module);

Mem::ConstantGlobalOp getNoneObject(mlir::ModuleOp& module);

constexpr std::string_view notImplementedTypeObjectName = "__builtins__.NotImplementedType";

Mem::ConstantGlobalOp getNotImplementedTypeObject(mlir::ModuleOp& module);

Mem::ConstantGlobalOp getNotImplementedObject(mlir::ModuleOp& module);

constexpr std::string_view tupleTypeObjectName = "__builtins__.tuple";

Mem::ConstantGlobalOp getTupleTypeObject(mlir::ModuleOp& module);

constexpr std::string_view stringTypeObjectName = "__builtins__.str";

Mem::ConstantGlobalOp getStringTypeObject(mlir::ModuleOp& module);

constexpr std::string_view boolTypeObjectName = "__builtins__.bool";

Mem::ConstantGlobalOp getBoolTypeObject(mlir::ModuleOp& module);

mlir::FunctionType getCCFuncType(mlir::MLIRContext* context);
} // namespace pylir::Mem
