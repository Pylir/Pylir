// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
