// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TYPE_FLOW_PASSES
#define TYPE_FLOW_PASSES

include "mlir/Pass/PassBase.td"

def TypeFlowLoopLift : Pass<"type-flow-loop-lift", "::pylir::TypeFlow::FuncOp">;

#endif
