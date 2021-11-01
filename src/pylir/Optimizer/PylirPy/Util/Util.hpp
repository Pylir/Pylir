#pragma once

#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Support/Macros.hpp>

namespace pylir::Py
{
mlir::Value buildException(mlir::Location loc, mlir::OpBuilder& builder, std::string_view kind,
                           std::vector<Py::IterArg> args, mlir::Block* PYLIR_NULLABLE exceptionPath);

mlir::Value buildCall(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value callable, mlir::Value tuple,
                      mlir::Value dict, mlir::Block* PYLIR_NULLABLE exceptionPath);

mlir::Value buildSpecialMethodCall(mlir::Location loc, mlir::OpBuilder& builder, llvm::Twine methodName,
                                   mlir::Value type, mlir::Value tuple, mlir::Value dict,
                                   mlir::Block* PYLIR_NULLABLE exceptionPath);
} // namespace pylir::Py
