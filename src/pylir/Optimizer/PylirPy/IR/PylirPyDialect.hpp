
#pragma once

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/IR/Dialect.h>

#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsDialect.h.inc"

namespace pylir::Py
{
constexpr llvm::StringLiteral specializationOfAttr = "py.specialization_of";

constexpr llvm::StringLiteral specializationTypeAttr = "py.specialization_args";

constexpr llvm::StringLiteral alwaysBoundAttr = "py.always_bound";
} // namespace pylir::Py
