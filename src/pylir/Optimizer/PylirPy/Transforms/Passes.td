// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIRPY_TRANSFORM_PASSES
#define PYLIRPY_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def ExpandPyDialectPass : Pass<"pylir-expand-py-dialect"> {
    let summary = "Expands ops of the Py dialect to operations that can be lowered";

    let dependentDialects = ["::pylir::Py::PylirPyDialect",
                             "::mlir::arith::ArithmeticDialect",
                             "::mlir::cf::ControlFlowDialect"];
}

def FoldHandlesPass : Pass<"pylir-fold-handles", "::mlir::ModuleOp"> {
    let summary = "Fold py.globalHandle";

    let dependentDialects = ["::pylir::Py::PylirPyDialect"];

    let statistics = [
        Statistic<"m_noLoadHandlesRemoved", "No-load Handles removed",
            "Amount of handles that were removed as they were never loaded">,
        Statistic<"m_singleStoreHandlesConverted", "Single-store Handles converted",
            "Amount of handles converted to values as they only had a single constant store">,
        Statistic<"m_singleRegionHandlesConverted", "Single-region Handles converted",
                    "Amount of handles turned to SSA as they only occur in a single region">
    ];
}

def HandleLoadStoreEliminationPass : Pass<"pylir-handle-load-store-elimination"> {
    let summary = "Eliminate loads and stores of handles";

    let statistics = [
        Statistic<"m_loadRemoved", "handle loads removed",
            "Amount of loads of handles that could be removed">,
        Statistic<"m_storesRemoved", "Stores to handles removed",
            "Amount of stores to handles removed">
    ];
}

def MonomorphPass : Pass<"pylir-monomorph", "::mlir::ModuleOp"> {
	let summary = "Monomorph module";

	let dependentDialects = ["::pylir::Py::PylirPyDialect", "::pylir::TypeFlow::TypeFlowDialect"];

	let statistics = [
		Statistic<"m_valuesReplaced", "Values replaced", "Amount of values that have been replaced with constants">,
		Statistic<"m_callsChanged", "Calls changed", "Amount of call instructions that were changed">,
		Statistic<"m_functionsCloned", "Function clones", "Amount of functions which have been cloned with more specific types">,
		Statistic<"m_typesOverDefined", "Types over defined", "Amount of types that have been marked over defined">,
	];

	let options = [
        Option<"m_maxTypeSize", "max-type-size", "std::size_t", "6",
                "Max amount of types a recursive type can have as function argument without being marked over defined">,
    ];
}

def TrialInlinerPass : Pass<"pylir-trial-inliner", "::mlir::ModuleOp"> {
	let summary = "Inline functions using trials";

	let dependentDialects = ["::pylir::Py::PylirPyDialect"];

	let statistics = [
		Statistic<"m_callsInlined", "Calls inlined", "Amount of call sites inlined">,
		Statistic<"m_cacheHits", "Cache hits", "Amount of times an inlining trial did not have to be performed due to a cache hit">,
		Statistic<"m_cacheMisses", "Caches misses", "Amount of times an inlining trial had to be performed as it was not in cache">,
		Statistic<"m_recursionLimitReached", "Recursion limits reached", "Amount of times a recursion chain was disabled in a caller">,
		Statistic<"m_optimizationRun", "Optimization pipeline run", "Amount of times the optimization pipeline had to be run">,
	];

	let options = [
		Option<"m_minCalleeSizeReduction", "min-callee-size-reduction", "std::uint32_t", "50",
			    "Percent the callee must have decreased in size after having been inlined in the caller to be deemed profitable">,
		Option<"m_maxRecursionDepth", "max-recursion-depth", "std::size_t", "4",
				"Maximum amount of times a recursive function deemed profitable may be inlined">,
        Option<"m_optimizationPipeline", "optimization-pipeline", "std::string",
               [{"canonicalize,pylir-sroa,canonicalize"}],
               "Optimization pipeline used to perform the inlining trials">,
	];
}

#endif
