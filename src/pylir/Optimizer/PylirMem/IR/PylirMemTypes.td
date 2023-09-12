// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_TYPES_TABLEGEN
#define PYLIR_TYPES_TABLEGEN

include "pylir/Optimizer/PylirMem/IR/PylirMemBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

class PylirMem_Type<string name, list<Trait> traits = [],
    string baseCppClass = "::mlir::Type"> :
  TypeDef<PylirMem_Dialect, name, traits, baseCppClass>;

def PylirMem_MemoryType : PylirMem_Type<"Memory"> {
  let mnemonic = "memory";
  let summary = "python memory type";
}

#endif
