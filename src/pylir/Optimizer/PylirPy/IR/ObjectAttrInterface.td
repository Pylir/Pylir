// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIRPY_INTERFACES_OBJECT_ATTR_INTERFACE
#define PYLIRPY_INTERFACES_OBJECT_ATTR_INTERFACE

include "mlir/IR/OpBase.td"
include "pylir/Optimizer/Util/TablegenUtil.td"

def ObjectAttrInterface : UseableAttrInterface<"ObjectAttrInterface"> {
    let cppNamespace = "::pylir::Py";

    let methods = [
        InterfaceMethod<[{}], "::mlir::FlatSymbolRefAttr", "getTypeObject", (ins)>,
        InterfaceMethod<[{}], "::mlir::DictionaryAttr", "getSlots", (ins)>
    ];
}

#endif
