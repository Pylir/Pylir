// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_PY_INTERFACES_READS_VALUE_ONLY
#define PYLIR_PY_INTERFACES_READS_VALUE_ONLY

include "mlir/IR/OpBase.td"
include "pylir/Optimizer/Util/OpVariableBase.td"

def OnlyReadsValueInterface : OpInterface<"OnlyReadsValueInterface"> {
  let cppNamespace = "::pylir::Py";

  // TODO: Reconsider warning below. Not sure where it'd be profitable for
  //       mutable objects, but might still want to allow it.
  //       CopyObjectInterface should instead have knowledge whether it is
  //       copying a mutable or immutable object.
  let description = [{
    Warning: This interface may only be attached to operations operating on
    immutable objects.

    This interface signifies that specific operands of an operation only read
    the value part of an operand.
    The value part of an operand is defined by everything part of the memory of
    the object that is:
    * NOT the type object
    * NOT any slots

    For user created objects not deriving from any builtin types but `object`,
    it is therefore nonsensical.
    For builtin types such as `int` or `str` however, it signifies reading the
    memory used to represent their value.

    To use this interface, simply attach `OnlyReadsValue` to operands using
    `Arg`.
  }];

  let methods = [
    ExtendableInterfaceMethod<[{

    }], "bool", "onlyReadsValue", (ins "::mlir::OpOperand&":$operand), [{
      bool result = false;
    }], "return result;">
  ];
}

def OnlyReadsValue : OpVariableGen<OnlyReadsValueInterface> {
  let methodGens = [
    OpVariableMethodGen<"onlyReadsValue", [{
      result = result || &$_arg == &operand;
    }]>
  ];
}

#endif
