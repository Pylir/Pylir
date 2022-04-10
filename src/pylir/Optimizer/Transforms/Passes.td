
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
