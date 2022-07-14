
// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <pylir/Optimizer/PylirPy/Analysis/TypeFlowIR.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>

#include "Passes.hpp"

namespace
{
#define GEN_PASS_CLASSES
#include <pylir/Optimizer/PylirPy/Transforms/Passes.h.inc>
} // namespace
