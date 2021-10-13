
#pragma once

#include <mlir/Pass/Pass.h>

#include <memory>

namespace pylir::Py
{
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createExpandPyDialectPass();

}
