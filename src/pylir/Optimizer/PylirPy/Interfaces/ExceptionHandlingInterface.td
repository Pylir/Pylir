#ifndef PYLIR_PY_INTERFACES_EXCEPTION_HANDLING
#define PYLIR_PY_INTERFACES_EXCEPTION_HANDLING

include "mlir/IR/OpBase.td"

def ExceptionHandlingInterface : OpInterface<"ExceptionHandlingInterface"> {
    let cppNamespace = "::pylir::Py";

    let methods = [
        InterfaceMethod<[{}], "::mlir::Block*", "getHappyPath", (ins)>,
        InterfaceMethod<[{}], "::mlir::Block*", "getExceptionPath", (ins)>,
        InterfaceMethod<[{}], "::mlir::MutableOperandRange", "getNormalDestOperandsMutable", (ins)>,
        InterfaceMethod<[{}], "::mlir::MutableOperandRange", "getUnwindDestOperandsMutable", (ins)>,
        InterfaceMethod<[{}], "::mlir::Operation*", "cloneWithoutExceptionHandling", (ins "::mlir::OpBuilder&":$builder)>,
    ];
}

def AddableExceptionHandlingInterface : OpInterface<"AddableExceptionHandlingInterface"> {
    let cppNamespace = "::pylir::Py";

    let methods = [
        InterfaceMethod<[{}], "::mlir::Operation*", "cloneWithExceptionHandling",
            (ins "::mlir::OpBuilder&":$builder, "::mlir::Block*":$happy_path, "::mlir::Block*":$exception_path,
                 "::mlir::ValueRange":$unwind_operands)>,
    ];
}

#endif
