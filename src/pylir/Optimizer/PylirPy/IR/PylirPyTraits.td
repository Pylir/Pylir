// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_PY_TRAITS_TABLEGEN
#define PYLIR_PY_TRAITS_TABLEGEN

include "pylir/Optimizer/PylirPy/Interfaces/ObjectFromTypeObjectInterface.td"

/// Signifies that the results of the operation it is bound to are always bound. If such an operation returns an unbound
/// result, it is undefined behaviour.
def AlwaysBound : NativeOpTrait<"AlwaysBound"> {
    let cppNamespace = "pylir::Py";
}

/// Signifies that the results of the operation returns a new immutable object. It therefore has distinct object
/// identity from every other object marked 'ReturnsImmutable' or a 'MemAlloc' result.
def ReturnsImmutable : NativeOpTrait<"ReturnsImmutable"> {
    let cppNamespace = "pylir::Py";
}

/// Automatically implements 'TypeRefineableInterface' for an operation, making it always return the builtin type
/// 'type'.
class RefinedType<string type> : ParamNativeOpTrait<"RefinedType", "::pylir::Builtins::" # type> {
	let cppNamespace = "pylir::Py";
}

/// Automatically implements 'TypeRefineableInterface' for an operation, making it always return a tuple type as an
/// approximate result.
def RefinedTypeTupleApproximate : NativeOpTrait<"RefinedTypeTupleApproximate"> {
	let cppNamespace = "pylir::Py";
}

class RefinedObjectFromTypeObjectImpl : NativeOpTrait<"RefinedObjectFromTypeObjectImpl"> {
	let cppNamespace = "pylir::Py";
}

/// Automatically implements both 'ObjectFromTypeObjectInterface' and 'TypeRefineableInterface'. The former
/// assumes that operation has an ODS operand called '$type_object', which the later interface uses and returns in its
/// implementation.
def RefinedObjectFromTypeObject : TraitList<[RefinedObjectFromTypeObjectImpl<>, ObjectFromTypeObjectInterface]>;

#endif
