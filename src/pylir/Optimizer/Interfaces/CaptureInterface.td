#ifndef PYLIR_INTERFACES_CAPTURE_INTERFACE
#define PYLIR_INTERFACES_CAPTURE_INTERFACE

include "mlir/IR/OpBase.td"

def CaptureInterface : OpInterface<"CaptureInterface"> {
    let cppNamespace = "::pylir";

    let methods = [
        InterfaceMethod<[{}], "bool", "capturesOperand", (ins "unsigned":$operandIndex)>
    ];
}

#endif
