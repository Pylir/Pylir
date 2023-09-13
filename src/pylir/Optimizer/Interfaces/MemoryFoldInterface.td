// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_INTERFACES_MEMORY_FOLD_INTERFACE
#define PYLIR_INTERFACES_MEMORY_FOLD_INTERFACE

include "mlir/IR/OpBase.td"

def MemoryFoldInterface : OpInterface<"MemoryFoldInterface"> {
  let cppNamespace = "::pylir";

  let methods = [
    InterfaceMethod<[{

    }], "::mlir::LogicalResult", "foldUsage", (ins
      "mlir::Operation*":$lastClobber,
      "::llvm::SmallVectorImpl<::mlir::OpFoldResult>&":$results)>
  ];
}

#endif
