// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TEST_PASSES
#define TEST_PASSES

include "mlir/Pass/PassBase.td"

def TestMemorySSA : Pass<"test-memory-ssa","::mlir::ModuleOp"> {
    let constructor = "::createTestMemorySSA()";
    let dependentDialects = ["::pylir::MemSSA::MemorySSADialect"];
}

def TestInlinerInterface : Pass<"test-inliner-interface", "::mlir::ModuleOp"> {
	let constructor = "::createTestInlinerInterface()";
	let dependentDialects = ["::pylir::Py::PylirPyDialect"];
}

def TestTypeFlow : Pass<"test-type-flow", "::mlir::ModuleOp"> {
    let constructor = "::createTestTypeFlow()";
    let dependentDialects = ["::pylir::TypeFlow::TypeFlowDialect"];
}

def TestAliasSetTracker : Pass<"test-alias-set-tracker", "::mlir::ModuleOp"> {
    let constructor = "::createTestAliasSetTracker()";
}

def TestLoopInfo : Pass<"test-loop-info", "::mlir::ModuleOp"> {
    let constructor = "::createTestLoopInfo()";
}

def TestHelloWorld : Pass<"test-hello-world"> {
    let constructor = "::createTestHelloWorld()";
}

def TestLinker : Pass<"test-linker", "::mlir::ModuleOp"> {
    let constructor = "::createTestLinker()";
}

#endif
