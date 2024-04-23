// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_HIR_ENUMS_TABLEGEN
#define PYLIR_HIR_ENUMS_TABLEGEN

include "mlir/IR/EnumAttr.td"

def PylirHIR_BinaryOperationAttr : I32EnumAttr<"BinaryOperation",
  "Binary operations in python", [
  I32EnumAttrCase<"Eq", 0, "__eq__">,
  I32EnumAttrCase<"Ne", 1, "__ne__">,
  I32EnumAttrCase<"Lt", 2, "__lt__">,
  I32EnumAttrCase<"Le", 3, "__le__">,
  I32EnumAttrCase<"Gt", 4, "__gt__">,
  I32EnumAttrCase<"Ge", 5, "__ge__">,
  I32EnumAttrCase<"Add", 6, "__add__">,
  I32EnumAttrCase<"Sub", 7, "__sub__">,
  I32EnumAttrCase<"Or", 8, "__or__">,
  I32EnumAttrCase<"Xor", 9, "__xor__">,
  I32EnumAttrCase<"And", 10, "__and__">,
  I32EnumAttrCase<"LShift", 11, "__lshift__">,
  I32EnumAttrCase<"RShift", 12, "__rshift__">,
  I32EnumAttrCase<"Mul", 13, "__mul__">,
  I32EnumAttrCase<"Div", 14, "__div__">,
  I32EnumAttrCase<"FloorDiv", 15, "__floordiv__">,
  I32EnumAttrCase<"Mod", 16, "__mod__">,
  I32EnumAttrCase<"MatMul", 17, "__matmul__">,
]> {
  let cppNamespace = "pylir::HIR";
}

// Enums values kept in sync with 'PylirHIR_BinaryOperationAttr'.
def PylirHIR_BinaryAssignmentAttr : I32EnumAttr<"BinaryAssignment",
  "Binary assignment operations in python", [
  I32EnumAttrCase<"Add", 6, "__iadd__">,
  I32EnumAttrCase<"Sub", 7, "__isub__">,
  I32EnumAttrCase<"Or", 8, "__ior__">,
  I32EnumAttrCase<"Xor", 9, "__ixor__">,
  I32EnumAttrCase<"And", 10, "__iand__">,
  I32EnumAttrCase<"LShift", 11, "__ilshift__">,
  I32EnumAttrCase<"RShift", 12, "__irshift__">,
  I32EnumAttrCase<"Mul", 13, "__imul__">,
  I32EnumAttrCase<"Div", 14, "__idiv__">,
  I32EnumAttrCase<"FloorDiv", 15, "__ifloordiv__">,
  I32EnumAttrCase<"Mod", 16, "__imod__">,
  I32EnumAttrCase<"MatMul", 17, "__imatmul__">,
]> {
  let cppNamespace = "pylir::HIR";
}

#endif
