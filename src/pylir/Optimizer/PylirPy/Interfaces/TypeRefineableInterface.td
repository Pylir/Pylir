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
        InterfaceMethod<[{}], "::llvm::SmallVector<::pylir::Py::ObjectTypeInterface>", "refineTypes",
                                (ins "llvm::ArrayRef<pylir::Py::ObjectTypeInterface>":$inputTypes,
                                     "::mlir::SymbolTable*":$symbolTable)>
    ];
}

#endif
