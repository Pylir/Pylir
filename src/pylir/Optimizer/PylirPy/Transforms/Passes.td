
#ifndef PYLIRPY_TRANSFORM_PASSES
#define PYLIRPY_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def ExpandPyDialect : Pass<"expand-py-dialect", "::mlir::ModuleOp"> {
    let summary = "Expands ops of the Py dialect to operations that can be lowered";
    let constructor = "::pylir::Py::createExpandPyDialectPass()";
    let dependentDialects = ["::pylir::Py::PylirPyDialect", "::mlir::arith::ArithmeticDialect"];
}

def FoldHandles : Pass<"fold-handles", "::mlir::ModuleOp"> {
    let summary = "Fold py.globalHandle";
    let constructor = "::pylir::Py::createFoldHandlesPass()";
    let dependentDialects = ["::pylir::Py::PylirPyDialect"];

    let statistics = [
        Statistic<"m_noLoadHandlesRemoved", "No-load Handles removed",
            "Amount of handles that were removed as they were never loaded">,
        Statistic<"m_singleStoreHandlesConverted", "Single-store Handles converted",
            "Amount of handles converted to values as they only had a single constant store">
    ];
}

def HandleLoadStoreElimination : Pass<"handle-load-store-elimination"> {
    let summary = "Eliminate loads and stores of handles";
    let constructor = "::pylir::Py::createHandleLoadStoreEliminationPass()";

    let statistics = [
        Statistic<"m_loadRemoved", "handle loads removed",
            "Amount of loads of handles that could be removed">,
        Statistic<"m_storesRemoved", "Stores to handles removed",
            "Amount of stores to handles removed">
    ];
}

#endif
