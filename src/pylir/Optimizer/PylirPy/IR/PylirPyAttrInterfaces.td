// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIRPY_INTERFACES_OBJECT_ATTR_INTERFACE
#define PYLIRPY_INTERFACES_OBJECT_ATTR_INTERFACE

include "mlir/IR/OpBase.td"

def ObjectAttrInterface : AttrInterface<"ObjectAttrInterface"> {
    let cppNamespace = "::pylir::Py";

    let methods = [
        InterfaceMethod<[{}], "::pylir::Py::RefAttr", "getTypeObject", (ins)>,
        InterfaceMethod<[{}], "::mlir::DictionaryAttr", "getSlots", (ins)>
    ];
}

def IntAttrInterface : AttrInterface<"IntAttrInterface", [ObjectAttrInterface]> {
    let cppNamespace = "::pylir::Py";

    let description = [{
        Interface implemented by any attribute that subclasses `builtins.int`.
        Implies an implementation of `ObjectAttrInterface`.
    }];

    let methods = [
        InterfaceMethod<[{
            Returns the big integer value of this attribute.
        }],
        "pylir::BigInt", "getInteger", (ins)>
    ];

    let returnType = "pylir::BigInt";
    let convertFromStorage = "$_self.getInteger()";
}

def BoolAttrInterface : AttrInterface<"BoolAttrInterface", [IntAttrInterface]> {
    let cppNamespace = "::pylir::Py";

    let description = [{
        Interface implemented by any attribute that subclasses `builtins.bool`.

        Implies an implementation of `IntAttrInterface`.
    }];

    let methods = [
        InterfaceMethod<[{
            Returns the boolean value of this attribute.
        }],
        "bool", "getBoolean", (ins)>
    ];

    let returnType = "bool";
    let convertFromStorage = "$_self.getBoolean()";
}

#endif
