// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIRPY_INTERFACES_OBJECT_ATTR_INTERFACE
#define PYLIRPY_INTERFACES_OBJECT_ATTR_INTERFACE

include "mlir/IR/OpBase.td"

def ObjectAttrInterface : AttrInterface<"ObjectAttrInterface"> {
  let cppNamespace = "::pylir::Py";

  let description = [{
    This interface is the base interface used by all object attributes and
    interfaces.
    It represents the immutable properties of all python object attributes.
  }];

  let methods = [
    InterfaceMethod<[{
      Returns the `#py.ref` referring to the type object of this attribute.
      Mustn't be null.
    }], "::pylir::Py::RefAttr", "getTypeObject", (ins)>
  ];
}

def ConstObjectAttrInterface : AttrInterface<"ConstObjectAttrInterface", [ObjectAttrInterface]> {
  let cppNamespace = "::pylir::Py";

  let description = [{
    This interface is a specialization of `ObjectAttrInterface`, representing
    instances of constant python objects.
    Note that "constant" here refers to constant in the IR and compiler sense,
    as in, a known value that cannot change, not immutable in the Python sense
    (e.g. a tuple).
    This interface is therefore implemented by all concrete object attributes
    including attributes used to represent otherwise mutable objects as
    attributes such as `#py.dict`.

    The methods on this interface are intended to return data that would only
    be valid on a constant as its value may change throughout the program
    otherwise.
    Immutable data is accessible via the `ObjectAttrInterface`.

    Implies an implementation of `ObjectAttrInterface`.
  }];

  let methods = [
    InterfaceMethod<[{
      Returns a dictionary containing all slots of the attribute.
      Mustn't be null.
    }], "::mlir::DictionaryAttr", "getSlots", (ins)>
  ];
}

def IntAttrInterface : AttrInterface<"IntAttrInterface",
  [ObjectAttrInterface]> {
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
