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

            An operation may return one of three results:
            `TypeRefineResult::Failure` signals that no proper results could be calculated from the given inputs.
            `TypeRefineResult::Success` signals that an exact result was returned that is suitable for constant folding.
            `TypeRefineResult::Approximate` signals that only a conservative result could be returned. It is unsuitable
            for constant folding in `typeFlow` as a more accurate result may still be achieved during interpretation.
            The value returned has to be a conceptual supertype of all possible exact results. This makes it suitable
            for constant folding of `py.typeOf` eg, as a returned type object HAS to be equal to all exact results.

            If there is no failure, the types of the results of this op are expected to be pushed into `resultTypes`,
            which must be equal to the amount of results of the operation at the end of the call.
        }], "::pylir::Py::TypeRefineResult", "refineTypes",
                                (ins "::llvm::ArrayRef<::pylir::Py::TypeAttrUnion>":$inputs,
                                     "::llvm::SmallVectorImpl<::pylir::Py::ObjectTypeInterface>&":$result)>
    ];
}

#endif
