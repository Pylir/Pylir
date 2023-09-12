// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_PY_INTERFACES_COPY_OBJECT
#define PYLIR_PY_INTERFACES_COPY_OBJECT

include "mlir/IR/OpBase.td"
include "pylir/Optimizer/Util/OpVariableBase.td"

def CopyObjectInterface : OpInterface<"CopyObjectInterface"> {
  let cppNamespace = "::pylir::Py";

  let methods = [
    InterfaceMethod<[{}], "::mlir::Value", "getTypeObject", (ins), "", "">,
    ExtendableInterfaceMethod<[{

    }], "::mlir::OpOperand&", "getCopiedOperand", (ins), "",
    "PYLIR_UNREACHABLE;">
  ];

  let description = [{
    This interface signifies that an operation does a copy operation of a
    builtin.
    The resulting object may have a new type which must be a subtype of the
    builtin type the operation operates on.

    By default, this interface expects the type object for the result to be an
    ODS operand called `$type_object`. The operand that is being copied should
    be marked via `Copied`.

    Note: If you are applying this interface you likely also want:
    `ObjectFromTypeObjectInterface`, `MemAlloc` on the result,
    `RefinedObjectFromTypeObject` and `OnlyReadsValueInterface`.
    You may also want `AlwaysBound`.
  }];

  let verify = [{
    static_assert(ConcreteOp::template hasTrait<::mlir::OpTrait::OneResult>());
    return mlir::success();
  }];
}

def Copied : OpVariableGen<CopyObjectInterface> {
  let methodGens = [
    OpVariableMethodGen<"getCopiedOperand", [{
      return $_arg;
    }]>
  ];
}

#endif
