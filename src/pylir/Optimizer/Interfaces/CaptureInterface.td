// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
