// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/ArrayRef.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyTypes.hpp>
#include <pylir/Optimizer/PylirPy/IR/TypeRefineableInterface.hpp>

namespace pylir::TypeFlow
{
using OpFoldResult = llvm::PointerUnion<Py::TypeAttrUnion, mlir::Value>;
} // namespace pylir::TypeFlow

#include "pylir/Optimizer/PylirPy/Analysis/TypeFlowExecInterface.h.inc"
