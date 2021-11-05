
#ifndef PYLIRPY_TRANSFORM_PASSES
#define PYLIRPY_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def ExpandPyDialect : FunctionPass<"expand-py-dialect"> {
    let summary = "Expands ops of the Py dialect to operations that can be lowered";
    let constructor = "::pylir::Py::createExpandPyDialectPass()";
    let dependentDialects = ["::pylir::Py::PylirPyDialect", "::mlir::StandardOpsDialect"];
}

def RemoveAllocas : FunctionPass<"remove-allocas"> {
    let summary = "Removes all instances of py.alloca and associated loads and stores";
    let constructor = "::pylir::Py::createRemoveAllocasPass()";
    let dependentDialects = ["::pylir::Py::PylirPyDialect"];
}

#endif
