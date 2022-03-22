
#ifndef PYLIRPY_TRANSFORM_PASSES
#define PYLIRPY_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def ExpandPyDialect : Pass<"expand-py-dialect", "::mlir::ModuleOp"> {
    let summary = "Expands ops of the Py dialect to operations that can be lowered";
    let constructor = "::pylir::Py::createExpandPyDialectPass()";
    let dependentDialects = ["::pylir::Py::PylirPyDialect",
                             "::mlir::arith::ArithmeticDialect",
                             "::mlir::cf::ControlFlowDialect",
                             "::mlir::func::FuncDialect"];
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

def Monomorph : Pass<"monomorph", "::mlir::ModuleOp"> {
	let summary = "Monomorph module";
	let constructor = "::pylir::Py::createMonomorphPass()";
	let dependentDialects = ["::pylir::Py::PylirPyDialect"];

	let statistics = [
		Statistic<"m_typesRefined", "Types refined", "Amount of types of operations which have been made more specific">,
		Statistic<"m_functionsCloned", "Function clones", "Amount of functions which have been cloned with more specified types">,
	];
}

#endif
