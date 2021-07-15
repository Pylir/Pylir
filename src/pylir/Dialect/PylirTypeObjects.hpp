
#pragma once

#include "PylirOps.hpp"

namespace pylir::Dialect
{
Dialect::GlobalOp getTypeTypeObject(mlir::ModuleOp& module);

Dialect::GlobalOp getFunctionTypeObject(mlir::ModuleOp& module);

mlir::FunctionType getCCFuncType(mlir::MLIRContext* context);
} // namespace pylir::Dialect
