// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_BASE_TABLEGEN
#define PYLIR_BASE_TABLEGEN

include "mlir/IR/OpBase.td"

def PylirMem_Dialect : Dialect {
  let name = "pyMem";
  let summary = "A dialect representing python programs";
  let description = [{
    TODO
  }];
  let cppNamespace = "::pylir::Mem";

  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif
