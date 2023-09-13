// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_HIR_BASE_TABLEGEN
#define PYLIR_HIR_BASE_TABLEGEN

include "mlir/IR/OpBase.td"

def PylirHIR_Dialect : Dialect {
  let name = "pyHIR";

  let description = [{
    Dialect for a high level representation of python operations.
    This dialect is meant as an entry point into MLIR and as the lowering target
    from the frontend.
    It aims to preserve the high level semantics and operations present in
    python source code and only later lower to
    "python mandated implementation details" operating on low level object
    operations (see {doc}`./PylirPy`).
  }];

  let cppNamespace = "::pylir::HIR";
  let dependentDialects = ["::pylir::Py::PylirPyDialect"];
  let usePropertiesForAttributes = 1;
}

#endif
