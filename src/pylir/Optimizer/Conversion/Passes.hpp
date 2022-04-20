// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Pass/Pass.h>

#include <llvm/ADT/Triple.h>
#include <llvm/IR/DataLayout.h>

#include <memory>

namespace pylir
{
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createConvertPylirPyToPylirMemPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createConvertPylirToLLVMPass(llvm::Triple targetTriple,
                                                                                  const llvm::DataLayout& dataLayout);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createConvertPylirToLLVMPass();

} // namespace pylir

namespace pylir
{
#define GEN_PASS_REGISTRATION
#include "pylir/Optimizer/Conversion/Passes.h.inc"

} // namespace pylir
