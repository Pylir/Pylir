
#pragma once

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/Pass.h>

#include <memory>

namespace pylir::Py
{
std::unique_ptr<mlir::Pass> createExpandPyDialectPass();

std::unique_ptr<mlir::Pass> createFoldHandlesPass();

std::unique_ptr<mlir::Pass> createHandleLoadStoreEliminationPass();

std::unique_ptr<mlir::Pass> createMonomorphPass();

std::unique_ptr<mlir::Pass> createInlinerPass();

#define GEN_PASS_REGISTRATION
#include "pylir/Optimizer/PylirPy/Transforms/Passes.h.inc"

} // namespace pylir::Py
