// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIRPY_ENUMS_TABLEGEN
#define PYLIRPY_ENUMS_TABLEGEN

include "mlir/IR/EnumAttr.td"

def PylirPy_IntCmpEq : I64EnumAttrCase<"eq", 0>;
def PylirPy_IntCmpNe : I64EnumAttrCase<"ne", 1>;
def PylirPy_IntCmpLt : I64EnumAttrCase<"lt", 2>;
def PylirPy_IntCmpLe : I64EnumAttrCase<"le", 3>;
def PylirPy_IntCmpGt : I64EnumAttrCase<"gt", 4>;
def PylirPy_IntCmpGe : I64EnumAttrCase<"ge", 5>;

def PylirPy_IntCmpKindAttr : I64EnumAttr<"IntCmpKind", "", [
  PylirPy_IntCmpEq, PylirPy_IntCmpNe, PylirPy_IntCmpLt, PylirPy_IntCmpLe,
  PylirPy_IntCmpGt, PylirPy_IntCmpGe
]> {
  let cppNamespace = "::pylir::Py";
}

#endif
