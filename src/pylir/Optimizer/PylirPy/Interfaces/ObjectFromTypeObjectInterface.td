#ifndef PYLIR_PY_INTERFACES_OBJECT_FROM_TYPE_OBJECT
#define PYLIR_PY_INTERFACES_OBJECT_FROM_TYPE_OBJECT

include "mlir/IR/OpBase.td"

def ObjectFromTypeObjectInterface : OpInterface<"ObjectFromTypeObjectInterface"> {
    let cppNamespace = "::pylir::Py";

    let methods = [
        InterfaceMethod<[{}], "::mlir::Value", "getTypeObject", (ins)>,
    ];
}

#endif
