// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_PY_TRAITS_TABLEGEN
#define PYLIR_PY_TRAITS_TABLEGEN

include "pylir/Optimizer/PylirPy/Interfaces/ObjectFromTypeObjectInterface.td"
include "mlir/IR/AttrTypeBase.td"

class PylirPyOpTrait<string name> : NativeOpTrait<name> {
    let cppNamespace = "pylir::Py";
}

class PylirPyAttrTrait<string name> : NativeAttrTrait<name> {
    let cppNamespace = "pylir::Py";
}

class PylirPyParamOpTrait<string name, string param> : ParamNativeOpTrait<name, param> {
    let cppNamespace = "pylir::Py";
}

/// Signifies that the results of the operation it is bound to are always bound. If such an operation returns an unbound
/// result, it is undefined behaviour.
def AlwaysBound : PylirPyOpTrait<"AlwaysBound">;

/// Signifies that the results of the operation returns a new immutable object. It therefore has distinct object
/// identity from every other object marked 'ReturnsImmutable' or a 'MemAlloc' result.
def ReturnsImmutable : PylirPyOpTrait<"ReturnsImmutable">;

/// Automatically implements 'TypeRefineableInterface' for an operation, making it always return the builtin type
/// 'type'.
class RefinedType<string type> : PylirPyParamOpTrait<"RefinedType", "::pylir::Builtins::" # type>;

/// Automatically implements 'TypeRefineableInterface' for an operation, making it always return a tuple type as an
/// approximate result.
def RefinedTypeTupleApproximate : PylirPyOpTrait<"RefinedTypeTupleApproximate">;

/// Automatically implements both 'ObjectFromTypeObjectInterface' and 'TypeRefineableInterface'. The former
/// assumes that operation has an ODS operand called '$type_object', which the later interface uses and returns in its
/// implementation.
def RefinedObjectFromTypeObject : TraitList<[PylirPyOpTrait<"RefinedObjectFromTypeObjectImpl">,
                                             ObjectFromTypeObjectInterface]>;

/// Indicates that an attribute is of an immutable python object. This is mainly useful for the initializer of
/// 'py.globalValue'.
def ImmutableAttr : PylirPyAttrTrait<"ImmutableAttr">;

/// Op trait used to indicate that an exception handling variant of the op exists.
/// Adding this trait makes the operation automatically implement 'AddableExceptionHandlingInterface'.
/// The trait has to know for each operand whether it is a singular operand or variadic.
/// By default it tries to deduce this information from traits, but if it is incapable of doing so, one has to supply an
/// array of strings, containing the strings "Single" or "Variadic" for every ith operand of the op.
class AddableExceptionHandling<string clazz, list<string> shape = []> : ParamNativeOpTrait<"AddableExceptionHandling",
	"::pylir::Py::" # clazz # !foldl("", shape, acc, var, !strconcat(acc, ", ::pylir::Py::OperandShape::" # var))> {
    let cppNamespace = "pylir::Py";
}

#endif
