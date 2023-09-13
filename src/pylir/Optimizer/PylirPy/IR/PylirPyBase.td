// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_PY_BASE_TABLEGEN
#define PYLIR_PY_BASE_TABLEGEN

include "mlir/IR/DialectBase.td"

def PylirPy_Dialect : Dialect {
  let name = "py";
  let summary = "A dialect representing python programs";
  let description = [{
    TODO
  }];
  let cppNamespace = "::pylir::Py";

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::cf::ControlFlowDialect"
  ];

  let hasConstantMaterializer = 1;
  let hasCanonicalizer = 1;

  let useDefaultAttributePrinterParser = 1;

  let useDefaultTypePrinterParser = 1;

  let usePropertiesForAttributes = 1;

  let extraClassDeclaration = [{
    void initializeAttributes();

    void initializeTypes();

    void initializeExternalModels();
  }];
}

#endif
