// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIRPY_TRANSFORM_PASSES
#define PYLIRPY_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def ExpandPyDialectPass : Pass<"pylir-expand-py-dialect"> {
  let summary = "Expands ops of the Py dialect to operations that can be lowered";

  let dependentDialects = ["::pylir::Py::PylirPyDialect",
               "::mlir::arith::ArithDialect",
               "::mlir::cf::ControlFlowDialect"];
}

def FoldGlobalsPass : Pass<"pylir-fold-globals", "::mlir::ModuleOp"> {
  let summary = "Fold py.global";

  let dependentDialects = ["::pylir::Py::PylirPyDialect"];

  let statistics = [
    Statistic<"m_noLoadGlobalsRemoved", "No-load Globals removed",
      "Amount of handles that were removed as they were never loaded">,
    Statistic<"m_singleStoreGlobalsConverted", "Single-store Globals converted",
      "Amount of globals converted to values as they only had a single constant store">,
    Statistic<"m_singleRegionGlobalsConverted", "Single-region Globals converted",
          "Amount of globals turned to SSA as they only occur in a single region">
  ];
}

def GlobalLoadStoreEliminationPass : Pass<"pylir-global-load-store-elimination"> {
  let summary = "Eliminate loads and stores of globals";

  let statistics = [
    Statistic<"m_loadRemoved", "global loads removed",
      "Amount of loads of globals that could be removed">,
    Statistic<"m_storesRemoved", "Stores to globals removed",
      "Amount of stores to globals removed">
  ];
}

def GlobalSROAPass : Pass<"pylir-global-sroa", "::mlir::ModuleOp"> {
  let dependentDialects = ["::pylir::Py::PylirPyDialect"];

  let statistics = [
    Statistic<"m_globalsSplit", "Globals split", "Global aggregates that have been split into their values">,
  ];
}

def InlinerPass : Pass<"pylir-inliner", "mlir::ModuleOp"> {
  let summary = "Inline functions";

  let dependentDialects = ["::pylir::Py::PylirPyDialect"];

  let statistics = [
  	Statistic<"m_callsInlined", "Calls inlined", "Amount of call sites inlined">,
  	Statistic<"m_optimizationRun", "Optimization pipeline run", "Amount of times the optimization pipeline was run">,
  	Statistic<"m_directRecursionsDiscarded", "Direct recursions discarded",
  	    "Amount of callsites not considered while inlining as they call themselves">,
  	Statistic<"m_callsitesTooExpensive", "Callsites considered too expensive to inline",
      	    "Amount of callsites whose inlining cost is greater than the threshold">,
    Statistic<"m_doneEarly", "No more inlining changes to be done",
          "1 if the pass stopped due to not wanting to inline any more callsites instead of having reached"
          " the inlining iteration limit.">,
  ];

  let options = [
    Option<"m_maxInliningIterations", "max-inlining-iterations", "std::size_t", "8",
      			"Maximum amount of times the inliner should attempt to inline inbetween optimizations">,
    Option<"m_threshold", "threshold", "std::uint32_t", "250",
        "Cost threshold in abstract units allowing a call-site to still be inlined. "
        "Call-sites more expensive than the threshold are not inlined">,
    Option<"m_optimizationPipeline", "optimization-pipeline", "std::string", [{"any()"}],
         "Optimization pipeline interleaved between inlining">,
    Option<"m_cyclePenalty", "cycle-penalty", "std::uint32_t", "50",
      "Penalty in abstract units, applied to the cost of a call-site for each occurrence of a repeated inlining "
      "of a callable, through that call-site">,
  ];
}

#endif
