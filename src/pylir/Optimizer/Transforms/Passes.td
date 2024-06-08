// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_TRANSFORM_PASSES
#define PYLIR_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def LoadForwardingPass : Pass<"pylir-load-forwarding"> {
  let summary = "Forward results of stores to loads";

  let dependentDialects = ["::pylir::MemSSA::MemorySSADialect"];

  let statistics = [
    Statistic<"m_localLoadsReplaced", "Load instructions replaced",
      "Amount of load instructions replaced">,
  ];
}

def SROAPass  : Pass<"pylir-sroa"> {
  let summary = "Scalar Replacement Of Aggregates";

  let statistics = [
    Statistic<"m_aggregatesRemoved", "Aggregates removed",
      "Amount of aggregates that were removed">,
    Statistic<"m_readWriteOpsRemoved", "Aggregate read/write ops removed",
      "Amount of read/write ops of aggregates that were removed">,
  ];
}

def FixpointPass : Pass<"pylir-fixpoint"> {
  let summary = "Run optimization pipeline until fixpoint";

  let statistics = [
    Statistic<"m_maxIterationReached", "Max iterations reached",
      "Amount of times the maximum iteration count was reached before a fixpoint">,
  ];

  let options = [
    Option<"m_optimizationPipeline", "optimization-pipeline", "std::string", [{""}],
      "Optimization pipeline to run until fixpoint">,
    Option<"m_maxIterationCount", "max-iteration-count", "std::size_t", "10",
      "Maximum amount of iterations if no fixpoint is reached">,
  ];
}

def ConditionalsImplicationsPass : Pass<"pylir-conditionals-implications"> {
  let summary = "Simplify boolean expressions based on conditional branch paths";

  let statistics = [
    Statistic<"m_simplificationsMade", "Number of simplifications made",
      "Amount of simplification patterns that could be applied through the implications">,
  ];
}

def DeadCodeEliminationPass : Pass<"pylir-dce"> {
  let summary = "Remove blocks without predecessors";

  let statistics = [
    Statistic<"m_blocksRemoved", "Number of blocks removed",
      "Number of blocks that were removed as they no longer had any predecessors">
  ];
}

#endif
