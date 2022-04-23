// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_TYPE_FLOW_TABLEGEN
#define PYLIR_TYPE_FLOW_TABLEGEN

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def TypeFlow_Dialect : Dialect {
    let name = "typeFlow";
    let cppNamespace = "::pylir::TypeFlow";

    let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

class TypeFlow_Type<string name, list<Trait> traits = [], string baseCppClass = "::mlir::Type">
    : TypeDef<TypeFlow_Dialect, name, traits, baseCppClass>;

class TypeFlow_Attr<string name, list<Trait> traits = [], string baseCppClass = "::mlir::Attribute">
    : AttrDef<TypeFlow_Dialect, name, traits, baseCppClass>;

class TypeFlow_Op<string mnemonic, list<Trait> traits = []>
    : Op<TypeFlow_Dialect, mnemonic, traits>;

#endif
