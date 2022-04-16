
#ifndef PYLIRPY_TRANSFORM_PASSES
#define PYLIRPY_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def ExpandPyDialect : Pass<"pylir-expand-py-dialect", "::mlir::ModuleOp"> {
    let summary = "Expands ops of the Py dialect to operations that can be lowered";
    let constructor = "::pylir::Py::createExpandPyDialectPass()";
    let dependentDialects = ["::pylir::Py::PylirPyDialect",
                             "::mlir::arith::ArithmeticDialect",
                             "::mlir::func::FuncDialect"];
}

def FoldHandles : Pass<"pylir-fold-handles", "::mlir::ModuleOp"> {
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

def HandleLoadStoreElimination : Pass<"pylir-handle-load-store-elimination"> {
    let summary = "Eliminate loads and stores of handles";
    let constructor = "::pylir::Py::createHandleLoadStoreEliminationPass()";

    let statistics = [
        Statistic<"m_loadRemoved", "handle loads removed",
            "Amount of loads of handles that could be removed">,
        Statistic<"m_storesRemoved", "Stores to handles removed",
            "Amount of stores to handles removed">
    ];
}

def Monomorph : Pass<"pylir-monomorph", "::mlir::ModuleOp"> {
	let summary = "Monomorph module";
	let constructor = "::pylir::Py::createMonomorphPass()";
	let dependentDialects = ["::pylir::Py::PylirPyDialect"];

	let statistics = [
		Statistic<"m_operationsReplaced", "Operations replaced", "Amount of operations that have been replaced during folding">,
		Statistic<"m_callsTurnedStatic", "Calls turned static", "Amount of call instructions that are now statically resolved">,
		Statistic<"m_functionsCloned", "Function clones", "Amount of functions which have been cloned with more specified types">,
	];
}

def Inliner : Pass<"pylir-inliner", "::mlir::ModuleOp"> {
	let summary = "Inline functions";
	let constructor = "::pylir::Py::createInlinerPass()";
	let dependentDialects = ["::pylir::Py::PylirPyDialect"];

	let statistics = [
        Statistic<"m_callsInlined", "Calls inlined", "Amount of call sites inlined">,
	];

	let options = [
		Option<"m_maxFuncGrowth", "max-func-growth", "std::uint32_t", "150", "Percent a function is allowed to grow">,
		Option<"m_maxModuleGrowth", "max-module-growth", "std::uint32_t", "200", "Percent the module is allowed to grow">,
		Option<"m_maxRecursiveInlines", "max-recursive-inlines", "std::uint32_t", "4", "Amount of times a recursive function might be recursively inlined">,
	];
}

def TrialInliner : Pass<"pylir-trial-inliner", "::mlir::ModuleOp"> {
	let summary = "Inline functions using trials";
	let constructor = "::pylir::Py::createTrialInlinerPass()";
	let dependentDialects = ["::pylir::Py::PylirPyDialect"];

	let statistics = [
		Statistic<"m_callsInlined", "Calls inlined", "Amount of call sites inlined">,
		Statistic<"m_cacheHits", "Cache hits", "Amount of times an inlining trial did not have to be performed due to a cache hit">,
		Statistic<"m_cacheMisses", "Caches misses", "Amount of times an inlining trial had to be performed as it was not in cache">,
	];

	let options = [
		Option<"m_minCalleeSizeReduction", "min-callee-size-reduction", "std::uint32_t", "50",
			  "Percent the callee must have decreased in size after having been inlined in the caller to be deemed profitable">,
	];
}

#endif
