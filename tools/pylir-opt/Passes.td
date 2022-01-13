
#ifndef TEST_PASSES
#define TEST_PASSES

include "mlir/Pass/PassBase.td"

def TestMemorySSA : Pass<"test-memory-ssa","::mlir::ModuleOp"> {
    let constructor = "::createTestMemorySSA()";
    let dependentDialects = [];
}

#endif
