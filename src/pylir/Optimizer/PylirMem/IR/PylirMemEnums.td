// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_MEM_ENUMS_TABLEGEN
#define PYLIR_MEM_ENUMS_TABLEGEN

include "mlir/IR/EnumAttr.td"

/// Object layouts known to the compiler with a static size. May be replaced
/// with some kind of type descriptors if we ever introduce custom object
/// layouts as an optimization.
def PylirMem_LayoutTypeAttr : I32EnumAttr<"LayoutType",
  "Enum of types with known object layouts", [
  I32EnumAttrCase<"Object", 0, "object">,
  I32EnumAttrCase<"Type", 1, "type">,
  I32EnumAttrCase<"Float", 2, "float">,
  I32EnumAttrCase<"Function", 3, "function">,
  I32EnumAttrCase<"Tuple", 4, "tuple">,
  I32EnumAttrCase<"List", 5, "list">,
  I32EnumAttrCase<"String", 6, "string">,
  I32EnumAttrCase<"Dict", 7, "dict">,
  I32EnumAttrCase<"Int", 8, "int">,
  I32EnumAttrCase<"BaseException", 9>,
]> {
  let cppNamespace = "pylir::Mem";
}

#endif
