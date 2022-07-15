// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

std::unique_ptr<mlir::Pass> createTypeFlowMonomorphPass();

std::unique_ptr<mlir::Pass> createInlinerPass();

std::unique_ptr<mlir::Pass> createTrialInlinerPass();

std::unique_ptr<mlir::Pass> createSROAPass();

#define GEN_PASS_REGISTRATION
#include "pylir/Optimizer/PylirPy/Transforms/Passes.h.inc"

} // namespace pylir::Py
