
#pragma once

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/Pass.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>

namespace pylir::Py
{
#define GEN_PASS_CLASSES
#include <pylir/Optimizer/PylirPy/Transforms/Passes.h.inc>
} // namespace pylir::Py
