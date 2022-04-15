#ifndef PYLIRPY_INTERFACES_INT_ATTR_INTERFACE
#define PYLIRPY_INTERFACES_INT_ATTR_INTERFACE

include "mlir/IR/OpBase.td"
include "pylir/Optimizer/Util/TablegenUtil.td"

def IntAttrInterface : UseableAttrInterface<"IntAttrInterface"> {
    let cppNamespace = "::pylir::Py";

    let methods = [
        InterfaceMethod<"", "const ::pylir::BigInt&", "getIntegerValue", (ins)>
    ];

    let returnType = "::pylir::BigInt";
    let convertFromStorage = "$_self.getIntegerValue()";
}

#endif
