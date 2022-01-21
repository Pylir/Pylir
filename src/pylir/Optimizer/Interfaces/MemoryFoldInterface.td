#ifndef PYLIR_INTERFACES_MEMORY_FOLD_INTERFACE
#define PYLIR_INTERFACES_MEMORY_FOLD_INTERFACE

include "mlir/IR/OpBase.td"

def MemoryFoldInterface : OpInterface<"MemoryFoldInterface"> {
    let cppNamespace = "::pylir";

    let methods = [
        InterfaceMethod<[{}], "::mlir::LogicalResult", "foldUsage", (ins "mlir::Operation*":$lastClobber,
                                                            "::llvm::SmallVectorImpl<::mlir::OpFoldResult>&":$results)>
    ];
}

#endif
