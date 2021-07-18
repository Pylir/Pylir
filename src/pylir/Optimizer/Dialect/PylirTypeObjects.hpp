
#pragma once

#include "PylirOps.hpp"

namespace pylir::Dialect
{
constexpr std::string_view typeTypeObjectName = "pylir_type_type";

Dialect::GlobalOp getTypeTypeObject(mlir::ModuleOp& module);

constexpr std::string_view functionTypeObjectName = "pylir_function_type";

Dialect::GlobalOp getFunctionTypeObject(mlir::ModuleOp& module);

constexpr std::string_view longTypeObjectName = "pylir_long_type";

Dialect::GlobalOp getLongTypeObject(mlir::ModuleOp& module);

mlir::FunctionType getCCFuncType(mlir::MLIRContext* context);
} // namespace pylir::Dialect
