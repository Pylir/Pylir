// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TEST_PASSES
#define TEST_PASSES

include "mlir/Pass/PassBase.td"

def TestMemorySSAPass : Pass<"test-memory-ssa","::mlir::ModuleOp"> {
  let dependentDialects = ["::pylir::MemSSA::MemorySSADialect"];
}

def TestInlinerInterfacePass : Pass<"test-inliner-interface", "::mlir::ModuleOp"> {
	let dependentDialects = ["::pylir::Py::PylirPyDialect"];
}

def TestAliasSetTrackerPass : Pass<"test-alias-set-tracker", "::mlir::ModuleOp">;

def TestLoopInfoPass : Pass<"test-loop-info", "::mlir::ModuleOp">;

def TestHelloWorldPass : Pass<"test-hello-world">;

def TestLinkerPass : Pass<"test-linker", "::mlir::ModuleOp">;

def TestInlineAllPass : Pass<"test-inline-all", "::mlir::ModuleOp"> {

  let statistics = [
  	Statistic<"m_callsInlined", "Calls inlined", "Amount of call sites inlined">,
  	Statistic<"m_recursionLimitReached", "Recursion limits reached", "Amount of times a recursion chain was disabled in a caller">,
  ];

  let options = [
  	Option<"m_maxRecursionDepth", "max-recursion-depth", "std::size_t", "4",
  			"Maximum amount of times a recursive function deemed profitable may be inlined">,
    Option<"m_optimizationPipeline", "optimization-pipeline", "std::string", "",
        "Optimization pipeline used to perform the inlining trials">,
  ];
}

def TestAddChangePass : Pass<"test-add-change", "::mlir::ModuleOp"> {
  let dependentDialects = ["::pylir::test::TestDialect"];
}

def TestEscapeAnalysisPass : Pass<"test-escape-analysis", "::mlir::ModuleOp">;

#endif
