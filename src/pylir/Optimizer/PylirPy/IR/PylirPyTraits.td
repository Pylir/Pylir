// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_PY_TRAITS_TABLEGEN
#define PYLIR_PY_TRAITS_TABLEGEN

include "pylir/Optimizer/PylirPy/Interfaces/KnownTypeObjectInterface.td"
include "pylir/Optimizer/Util/TablegenUtil.td"
include "mlir/IR/AttrTypeBase.td"

class PylirPyOpTrait<string name> : NativeOpTrait<name> {
  let cppNamespace = "pylir::Py";
}

class PylirPyAttrTrait<string name> : NativeAttrTrait<name> {
  let cppNamespace = "pylir::Py";
}

class PylirPyParamOpTrait<string name, string param>
  : ParamNativeOpTrait<name, param> {
  let cppNamespace = "pylir::Py";
}

/// Signifies that the results of the operation it is bound to are always bound.
/// If such an operation returns an unbound result, it is undefined behaviour.
def AlwaysBound : PylirPyOpTrait<"AlwaysBound">;

/// Trait used to mark the entry blockarguments of all regions of an op as
/// always being bound.
def EntryArgsBound : PylirPyOpTrait<"EntryArgsBound">;

/// Signifies that the results of the operation returns a new immutable object.
/// It therefore has distinct object identity from every other object marked
/// 'ReturnsImmutable' or a 'MemAlloc' result.
def ReturnsImmutable : PylirPyOpTrait<"ReturnsImmutable">;

/// Automatically implements 'KnownTypeObjectInterface' for an operation, making
/// it always return the builtin type 'type'.
class KnownType<string type>
  : OpInterfaceImplementation<KnownTypeObjectInterface, [{
  ::mlir::OpFoldResult getKnownTypeObject();
}], [{
  ::mlir::OpFoldResult $cppClass::getKnownTypeObject() {
    return pylir::Py::GlobalValueAttr::get(getContext(), pylir::Builtins::}]
      # type # [{.name);
  }
}]>;

/// Op trait used to indicate that an exception handling variant of the op
/// exists. Adding this trait makes the operation automatically implement
/// 'AddableExceptionHandlingInterface'. The trait has to know for each operand
/// whether it is a singular operand or variadic. By default it tries to deduce
/// this information from traits, but if it is incapable of doing so, one has to
/// supply an array of strings, containing the strings "Single" or "Variadic"
/// for every ith operand of the op.
class AddableExceptionHandling<string clazz, list<string> shape = []>
  : ParamNativeOpTrait<"AddableExceptionHandling",
	    clazz # !foldl("", shape, acc, var,
	      !strconcat(acc, ", ::pylir::Py::OperandShape::" # var))> {
  let cppNamespace = "pylir::Py";
}

#endif
