// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_PY_TRAITS_TABLEGEN
#define PYLIR_PY_TRAITS_TABLEGEN

include "pylir/Optimizer/PylirPy/Interfaces/ObjectFromTypeObjectInterface.td"

def AlwaysBound : NativeOpTrait<"AlwaysBound"> {
    let cppNamespace = "pylir::Py";
}

def ReturnsImmutable : NativeOpTrait<"ReturnsImmutable"> {
    let cppNamespace = "pylir::Py";
}

def NoCapture : NativeOpTrait<"NoCapture"> {
    let cppNamespace = "pylir::Py";
}

class RefinedType<string type> : NativeOpTrait<type # "RefinedType"> {
	let cppNamespace = "pylir::Py";
}

def RefinedTypeTupleApproximate : NativeOpTrait<"RefinedTypeTupleApproximate"> {
	let cppNamespace = "pylir::Py";
}

def RefinedObjectFromTypeObjectImpl : NativeOpTrait<"RefinedObjectFromTypeObjectImpl"> {
	let cppNamespace = "pylir::Py";
}

def RefinedObjectFromTypeObject : TraitList<[RefinedObjectFromTypeObjectImpl, ObjectFromTypeObjectInterface]>;

#endif
