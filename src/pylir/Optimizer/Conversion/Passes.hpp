#pragma once

#include <mlir/Pass/Pass.h>

#include <llvm/ADT/Triple.h>
#include <llvm/IR/DataLayout.h>

#include <memory>

namespace pylir
{
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createConvertPylirPyToPylirMemPass();
}

namespace pylir::Mem
{
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createConvertPylirToLLVMPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createConvertPylirToLLVMPass(llvm::Triple targetTriple,
                                                                                  const llvm::DataLayout& dataLayout);

} // namespace pylir::Mem

namespace pylir
{
#define GEN_PASS_REGISTRATION
#include "pylir/Optimizer/Conversion/Passes.h.inc"

} // namespace pylir
