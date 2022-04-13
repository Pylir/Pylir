#ifndef PYLIR_PY_INTERFACES_TYPE_FOLD_INTERFACE
#define PYLIR_PY_INTERFACES_TYPE_FOLD_INTERFACE

include "mlir/IR/OpBase.td"

def TypeFoldInterface : OpInterface<"TypeFoldInterface"> {
    let cppNamespace = "::pylir::Py";

    let methods = [
        InterfaceMethod<[{}], "::mlir::OpFoldResult", "typeFold",
                                (ins "llvm::ArrayRef<pylir::Py::ObjectTypeInterface>":$inputTypes)>
    ];

    let verify = [{
        static_assert(ConcreteOp::template hasTrait<::mlir::OpTrait::OneResult>());
        return ::mlir::success();
    }];
}

#endif
