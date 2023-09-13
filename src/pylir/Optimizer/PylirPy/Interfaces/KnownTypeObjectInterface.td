// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_PY_INTERFACES_KNOWN_TYPE_OBJECT
#define PYLIR_PY_INTERFACES_KNOWN_TYPE_OBJECT

include "mlir/IR/OpBase.td"

def KnownTypeObjectInterface : OpInterface<"KnownTypeObjectInterface"> {
  let cppNamespace = "::pylir::Py";

  let methods = [
    InterfaceMethod<[{
      Returns the type object of the result of this operation or nullptr if
      unknown.
      The type object may either be a value or an attribute.
    }], "::mlir::OpFoldResult", "getKnownTypeObject", (ins), "", [{
      return $_op.getTypeObject();
    }]>,
  ];

  let verify = [{
    static_assert(ConcreteOp::template hasTrait<::mlir::OpTrait::OneResult>());
    return mlir::success();
  }];
}

#endif
