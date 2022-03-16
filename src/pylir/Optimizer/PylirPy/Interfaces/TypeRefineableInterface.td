#ifndef PYLIR_PY_INTERFACES_TYPE_REFINEABLE_INTERFACE
#define PYLIR_PY_INTERFACES_TYPE_REFINEABLE_INTERFACE

include "mlir/IR/OpBase.td"

def TypeRefineableInterface : OpInterface<"TypeRefineableInterface"> {
    let cppNamespace = "::pylir::Py";

    let methods = [
        InterfaceMethod<[{}], "::llvm::SmallVector<::pylir::Py::ObjectTypeInterface>", "refineTypes", (ins)>
    ];
}

#endif
