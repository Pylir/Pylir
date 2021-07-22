
#pragma once

#include "PylirOps.hpp"

namespace pylir::Dialect
{
constexpr std::string_view typeTypeObjectName = "__builtins__.type";

Dialect::ConstantGlobalOp getTypeTypeObject(mlir::ModuleOp& module);

constexpr std::string_view functionTypeObjectName = "__builtins__.function";

Dialect::ConstantGlobalOp getFunctionTypeObject(mlir::ModuleOp& module);

constexpr std::string_view longTypeObjectName = "__builtins__.long";

Dialect::ConstantGlobalOp getLongTypeObject(mlir::ModuleOp& module);

constexpr std::string_view noneTypeObjectName = "__builtins__.NoneType";

Dialect::ConstantGlobalOp getNoneTypeObject(mlir::ModuleOp& module);

Dialect::ConstantGlobalOp getNoneObject(mlir::ModuleOp& module);

constexpr std::string_view notImplementedTypeObjectName = "__builtins__.NotImplementedType";

Dialect::ConstantGlobalOp getNotImplementedTypeObject(mlir::ModuleOp& module);

Dialect::ConstantGlobalOp getNotImplementedObject(mlir::ModuleOp& module);

mlir::FunctionType getCCFuncType(mlir::MLIRContext* context);
} // namespace pylir::Dialect