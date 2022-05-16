// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <pylir/Optimizer/PylirPy/Interfaces/TypeFoldInterface.hpp>

#include "pylir/Optimizer/PylirPy/Analysis/TypeFlowIRDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "pylir/Optimizer/PylirPy/Analysis/TypeFlowIRAttributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "pylir/Optimizer/PylirPy/Analysis/TypeFlowIRTypes.h.inc"

#define GET_OP_CLASSES
#include "pylir/Optimizer/PylirPy/Analysis/TypeFlowIROps.h.inc"
