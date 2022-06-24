// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_PY_INTERFACES_TYPE_REFINEABLE_INTERFACE
#define PYLIR_PY_INTERFACES_TYPE_REFINEABLE_INTERFACE

include "mlir/IR/OpBase.td"

def TypeRefineableInterface : OpInterface<"TypeRefineableInterface"> {
    let cppNamespace = "::pylir::Py";

    let methods = [
        InterfaceMethod<[{
            This method attempts to returns the type of each result of this operation, given the known input types and
            values of each operand. `inputs` is an array that corresponds with the amount and order of operands, of the
            operation. For each operand of type `py.dynamic`, the input type is passed in, for all other operands of
            different types, constant values may be passed in.
            If the type or value of an operand is unknown, its corresponding array slot is a nullptr.
            This operation may not succeed in which case `TypeRefineResult::Failure` is returned.
            If only parts of the type can be deduced `TypeRefineResult::Approximate` is returned. This currently implies
            that a `ClassType` is returned instead of more concrete ObjectTypeInterface.
            All other cases should return `TypeRefineResult::Success`.

            If there is no failure, the types of the results of this op are expected to be pushed into `resultTypes`,
            which must be equal to the amount of results of the operation at the end of the call.
        }], "::pylir::Py::TypeRefineResult", "refineTypes",
                                (ins "::llvm::ArrayRef<::pylir::Py::TypeAttrUnion>":$inputs,
                                     "::llvm::SmallVectorImpl<::pylir::Py::ObjectTypeInterface>&":$resultTypes,
                                     "::mlir::SymbolTableCollection&":$symbolTableCollection)>
    ];
}

#endif
