#pragma once

#include <mlir/Pass/Pass.h>

#include <memory>

namespace pylir
{
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createConvertPylirPyToPylirMemPass();
}

namespace pylir::Mem
{
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createConvertPylirToLLVMPass();

}

namespace pylir
{
#define GEN_PASS_REGISTRATION
#include "pylir/Optimizer/Conversion/Passes.h.inc"

} // namespace pylir
