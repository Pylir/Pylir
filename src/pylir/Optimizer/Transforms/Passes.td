// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_TRANSFORM_PASSES
#define PYLIR_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def LoadForwarding : Pass<"pylir-load-forwarding"> {
    let summary = "Forward results of stores to loads";
    let constructor = "::pylir::createLoadForwardingPass()";
    let dependentDialects = ["::pylir::MemSSA::MemorySSADialect"];

    let statistics = [
        Statistic<"m_localLoadsReplaced", "Load instructions replaced", "Amount of load instructions replaced">,
    ];
}

#endif
