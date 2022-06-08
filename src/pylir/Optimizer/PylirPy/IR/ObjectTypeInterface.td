// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIRPY_INTERFACES_OBJECT_TYPE_INTERFACE
#define PYLIRPY_INTERFACES_OBJECT_TYPE_INTERFACE

include "mlir/IR/OpBase.td"

def ObjectTypeInterface : TypeInterface<"ObjectTypeInterface"> {
    let cppNamespace = "::pylir::Py";

    let methods = [
        InterfaceMethod<[{}], "::mlir::FlatSymbolRefAttr", "getTypeObject", (ins)>,
    ];
}

#endif
