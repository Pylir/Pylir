#pragma once

#include <mlir/Pass/Pass.h>

#include <memory>

namespace pylir::Mem
{
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createConvertPylirToLLVMPass();

}
