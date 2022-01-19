
#ifndef PYLIRPY_TRANSFORM_PASSES
#define PYLIRPY_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def ExpandPyDialect : Pass<"expand-py-dialect", "::mlir::ModuleOp"> {
    let summary = "Expands ops of the Py dialect to operations that can be lowered";
    let constructor = "::pylir::Py::createExpandPyDialectPass()";
    let dependentDialects = ["::pylir::Py::PylirPyDialect", "::mlir::StandardOpsDialect"];
}

def OptimizeHandles : Pass<"optimize-handles", "::mlir::ModuleOp"> {
    let summary = "Optimize usage of py.globalHandle";
    let constructor = "::pylir::Py::createOptimizeHandlesPass()";
    let dependentDialects = ["::pylir::Py::PylirPyDialect"];

    let statistics = [
        Statistic<"m_noLoadHandlesRemoved", "No-load Handles removed",
            "Amount of handles that were removed as they were never loaded">,
        Statistic<"m_singleStoreHandlesConverted", "Single-store Handles converted",
            "Amount of handles converted to values as they only had a single constant store">
    ];
}

#endif
