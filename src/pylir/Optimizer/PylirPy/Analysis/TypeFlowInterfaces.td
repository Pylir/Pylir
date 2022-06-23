// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_PY_INTERFACES_TYPE_FLOW
#define PYLIR_PY_INTERFACES_TYPE_FLOW

include "mlir/IR/OpBase.td"

def TypeFlowExecInterface : OpInterface<"TypeFlowExecInterface"> {
    let cppNamespace = "::pylir::TypeFlow";

    let methods = [
        InterfaceMethod<[{}], "::mlir::LogicalResult", "exec",
            (ins "::llvm::ArrayRef<::mlir::Attribute>":$operands,
                 "::llvm::SmallVectorImpl<::mlir::OpFoldResult>&":$results,
                 "::mlir::SymbolTableCollection&":$collection)>,
    ];
}

def TypeFlowValueMappingInterface : OpInterface<"TypeFlowValueMappingInterface"> {
    let cppNamespace = "::pylir::TypeFlow";

    let methods = [
        InterfaceMethod<[{
            Maps a result value of this operation to the corresponding value of the mapped instruction in the original
            IR.
        }], "::mlir::Value", "mapValue", (ins "::mlir::Value":$resultValue)>,
    ];
}

#endif
