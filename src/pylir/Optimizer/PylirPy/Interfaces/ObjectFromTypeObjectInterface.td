// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_PY_INTERFACES_OBJECT_FROM_TYPE_OBJECT
#define PYLIR_PY_INTERFACES_OBJECT_FROM_TYPE_OBJECT

include "mlir/IR/OpBase.td"

def ObjectFromTypeObjectInterface : OpInterface<"ObjectFromTypeObjectInterface"> {
    let cppNamespace = "::pylir::Py";

    let methods = [
        InterfaceMethod<[{}], "unsigned", "getTypeObjectIndex", (ins), "", [{
            return static_cast<mlir::OperandRange>($_op.getTypeObjectMutable()).getBeginOperandIndex();
        }]>,
        InterfaceMethod<[{}], "::mlir::Value", "getTypeObject", (ins), "", [{
            return $_op->getOperand($_op->getTypeObjectIndex());
        }]>,
    ];

    let verify = [{
        static_assert(ConcreteOp::template hasTrait<::mlir::OpTrait::OneResult>());
        return mlir::success();
    }];
}

#endif
