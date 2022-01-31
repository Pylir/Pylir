#ifndef PYLIRPY_INTERFACES_GET_TYPE_INTERFACE
#define PYLIRPY_INTERFACES_GET_TYPE_INTERFACE

include "mlir/IR/OpBase.td"

def RuntimeTypeInterface : OpInterface<"RuntimeTypeInterface"> {
    let cppNamespace = "::pylir::Py";

    let methods = [
        InterfaceMethod<[{}], "::mlir::OpFoldResult", "getRuntimeType", (ins "unsigned":$resultIndex)>
    ];
}

#endif
