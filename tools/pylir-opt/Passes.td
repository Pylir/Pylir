
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

#endif
