// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIRPY_INTERFACES_OBJECT_ATTR_INTERFACE
#define PYLIRPY_INTERFACES_OBJECT_ATTR_INTERFACE

include "mlir/IR/OpBase.td"

/// Base class for all attribute interfaces that should also be able to be
/// implementable by `GlobalValueAttr`. Any `GlobalValueAttr` that refers to a
/// symbol that implements a given interface automatically implements that
/// interface as well. If the attribute is a "sub-interface" of
/// `ConstObjectAttrInterface`, then the symbol referred to by the
/// `GlobalValueAttr` must additionally be constant.
class GlobalValueAttrImplementable<string name,
  list<Interface> baseInterfaces = []> : AttrInterface<name, baseInterfaces> {

  /// Interface method that should be appended to the methods array of any
  /// subclass of `GlobalValueAttrImplementable`. It automatically adds the
  /// necessary hooks to allow `GlobalValueAttr` to implement the interface.
  InterfaceMethod canImplementMethod = InterfaceMethod<[{
    This method is used by the `classof` mechanism of `}] # name # [{` to allow
    an implementation to dynamically determine whether it implements the
    interface.
    The implementation has a parameter containing `}] # name # [{` to avoid
    clashing with corresponding methods of any other interfaces implemented.

    Returns true for attributes `}] # name # [{` is directly attached to
    in ODS.
  }], "bool", "canImplement",
    (ins "std::in_place_type_t<" # cppNamespace # "::" # name # ">":$tagType),
     [{
    return true;
  }]>;

  let extraClassOf = [{
    return $_attr.canImplement(std::in_place_type<}]
      # cppNamespace # "::" # name # [{>);
  }];
}

def ObjectAttrInterface : GlobalValueAttrImplementable<"ObjectAttrInterface"> {
  let cppNamespace = "::pylir::Py";

  let description = [{
    This interface is the base interface used by all object attributes and
    interfaces.
    It represents the immutable properties of all python object attributes.
  }];

  let methods = [
    InterfaceMethod<[{
      Returns the attribute representing the type object of this attribute.
      Mustn't be null.
    }], "::pylir::Py::ObjectBaseAttribute", "getTypeObject", (ins), [{
      return $_attr.getTypeObject();
    }]>,
    canImplementMethod
  ];
}

def ConstObjectAttrInterface
  : GlobalValueAttrImplementable<"ConstObjectAttrInterface",
      [ObjectAttrInterface]> {
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
    }], "::mlir::DictionaryAttr", "getSlots", (ins)>,
    canImplementMethod
  ];
}

def IntAttrInterface : GlobalValueAttrImplementable<"IntAttrInterface",
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
    "pylir::BigInt", "getInteger", (ins)>,
    canImplementMethod
  ];

  let returnType = "pylir::BigInt";
  let convertFromStorage = "$_self.getInteger()";
}

def BoolAttrInterface : GlobalValueAttrImplementable<"BoolAttrInterface",
  [IntAttrInterface]> {
  let cppNamespace = "::pylir::Py";

  let description = [{
    Interface implemented by any attribute that subclasses `builtins.bool`.

    Implies an implementation of `IntAttrInterface`.
  }];

  let methods = [
    InterfaceMethod<[{
      Returns the boolean value of this attribute.
    }],
    "bool", "getBoolean", (ins)>,
    canImplementMethod
  ];

  let returnType = "bool";
  let convertFromStorage = "$_self.getBoolean()";
}

def TupleAttrInterface : GlobalValueAttrImplementable<"TupleAttrInterface",
  [ObjectAttrInterface]> {
  let cppNamespace = "::pylir::Py";

  let description = [{
    Interface implemented by any attribute that subclasses `builtins.tuple`.

    Implies an implementation of `ObjectAttrInterface`.
  }];

  let methods = [
    InterfaceMethod<[{
      Returns the elements of this tuple.
    }],
    "llvm::ArrayRef<mlir::Attribute>", "getElements", (ins)>,
    canImplementMethod
  ];

  let extraSharedClassDeclaration = [{
    [[nodiscard]] mlir::Attribute operator[](std::size_t index) const {
      return $_attr.getElements()[index];
    }

    /// Returns true if the tuple contains no elements.
    [[nodiscard]] bool empty() const {
      return $_attr.getElements().empty();
    }

    /// Returns the amount of elements within the tuple.
    [[nodiscard]] std::size_t size() const {
      return $_attr.getElements().size();
    }

    /// Returns the begin iterator to the first element in the tuple.
    [[nodiscard]] auto begin() const {
      return $_attr.getElements().begin();
    }

    /// Returns the end iterator past the last element in the tuple.
    [[nodiscard]] auto end() const {
      return $_attr.getElements().end();
    }
  }];
}

def TypeAttrInterface : GlobalValueAttrImplementable<"TypeAttrInterface",
  [ObjectAttrInterface]> {
  let cppNamespace = "::pylir::Py";

  let description = [{
    Interface implemented by any attribute that subclasses `builtins.type`.

    Implies an implementation of `ObjectAttrInterface`.
  }];

  let methods = [
    InterfaceMethod<[{
      Returns the MRO tuple of this type.
    }],
    "mlir::Attribute", "getMroTuple", (ins)>,
    InterfaceMethod<[{
      Returns the list of slots that an instance of this type contains.
      This must be a tuple of strings.
    }],
    "pylir::Py::TupleAttrInterface", "getInstanceSlots", (ins)>,
    canImplementMethod
  ];
}

def DictAttrInterface : GlobalValueAttrImplementable<"DictAttrInterface",
  [ConstObjectAttrInterface]> {
  let cppNamespace = "::pylir::Py";

  let description = [{
    Interface implemented by any attribute that subclasses `builtins.dict`.

    Implies an implementation of `ConstObjectAttrInterface`.
  }];

  let methods = [
    InterfaceMethod<[{
      Returns the value for a given key in the dictionary.
      Returns a null attribute if not found.
      The key must be a `RefAttr` or `ObjectAttrInterface`.
    }],
    "mlir::Attribute", "lookup", (ins "mlir::Attribute":$key)>,
    InterfaceMethod<[{
      Returns the list of key-value pairs of the dictionary in insertion order.
    }],
    "llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>>",
    "getKeyValuePairs", (ins)>,
    canImplementMethod
  ];
}

def FunctionAttrInterface
  : GlobalValueAttrImplementable<"FunctionAttrInterface",
    [ObjectAttrInterface]> {
  let cppNamespace = "::pylir::Py";

  let description = [{
    Interface implemented by any attribute that subclasses `builtins.function`.

    Implies an implementation of `ObjectAttrInterface`.
  }];

  let methods = [
    InterfaceMethod<[{
      Returns a symbol reference to the `py.func` this function attribute refers
      to.
    }],
    "mlir::FlatSymbolRefAttr", "getValue", (ins)>,
    canImplementMethod
  ];
}

def FloatAttrInterface : GlobalValueAttrImplementable<"FloatAttrInterface",
  [ObjectAttrInterface]> {
  let cppNamespace = "::pylir::Py";

  let description = [{
    Interface implemented by any attribute that subclasses `builtins.float`.

    Implies an implementation of `ObjectAttrInterface`.
  }];

  let methods = [
    InterfaceMethod<[{
      Returns double value of this attribute.
    }],
    "double", "getDoubleValue", (ins)>,
    canImplementMethod
  ];
}

def StrAttrInterface : GlobalValueAttrImplementable<"StrAttrInterface",
  [ObjectAttrInterface]> {
  let cppNamespace = "::pylir::Py";

  let description = [{
    Interface implemented by any attribute that subclasses `builtins.str`.

    Implies an implementation of `ObjectAttrInterface`.
  }];

  let methods = [
    InterfaceMethod<[{
      Returns string value of this attribute.
    }],
    "llvm::StringRef", "getValue", (ins)>,
    canImplementMethod
  ];
}

def EqualsAttrInterface
  : AttrInterface<"EqualsAttrInterface", [ObjectAttrInterface]> {
  let cppNamespace = "::pylir::Py";

  let description = [{
    Interface used to support equality as specified in Python in attributes.
  }];

  let methods = [
    InterfaceMethod<[{
      Returns the attribute acting as the canonical form for checking equality.
      The precise kind of attribute returned is unspecified.
      However, all instances of a python attribute that should compare equal
      must return the exact same attribute from this function.
      This makes it possible to compare the attributes returned by this function
      to check for equality in python semantics.

      Must not return null.
    }],
    "mlir::Attribute", "getCanonicalAttribute", (ins)>
  ];
}

#endif
