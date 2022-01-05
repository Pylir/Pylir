#pragma once

#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Support/Macros.hpp>

#include <string_view>

namespace pylir::Py
{
mlir::Value buildException(mlir::Location loc, mlir::OpBuilder& builder, std::string_view kind,
                           std::vector<Py::IterArg> args, mlir::Block* PYLIR_NULLABLE landingPadBlock);

mlir::Value buildSpecialMethodCall(mlir::Location loc, mlir::OpBuilder& builder, llvm::Twine methodName,
                                   mlir::Value tuple, mlir::Value /*nullable*/ kwargs,
                                   mlir::Block* PYLIR_NULLABLE exceptionPath,
                                   mlir::Block* PYLIR_NULLABLE landingPadBlock);

mlir::FunctionType getUniversalFunctionType(mlir::MLIRContext* context);

constexpr std::string_view pylirCallIntrinsic = "$pylir_call";

} // namespace pylir::Py
