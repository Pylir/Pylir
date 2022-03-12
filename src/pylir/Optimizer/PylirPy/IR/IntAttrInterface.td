#ifndef PYLIRPY_INTERFACES_INT_ATTR_INTERFACE
#define PYLIRPY_INTERFACES_INT_ATTR_INTERFACE

include "mlir/IR/OpBase.td"

def IntAttrInterface : AttrInterface<"IntAttrInterface"> {
    let cppNamespace = "::pylir::Py";

    let methods = [
        InterfaceMethod<"", "const ::pylir::BigInt&", "getIntegerValue", (ins)>
    ];
}

#endif
