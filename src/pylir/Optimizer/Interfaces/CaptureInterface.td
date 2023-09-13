// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_INTERFACES_CAPTURE_INTERFACE
#define PYLIR_INTERFACES_CAPTURE_INTERFACE

include "mlir/IR/OpBase.td"
include "pylir/Optimizer/Util/OpVariableBase.td"

def CaptureInterface : OpInterface<"CaptureInterface"> {
  let cppNamespace = "::pylir";

  let description = [{
    This interface allows operations to specify whether they capture an object
    reference.
    A capture is defined as copying the object reference itself, making it
    possible for it to persist longer than its SSA value.
    Operations that simply use an object reference as operand to read from are
    typical examples for operations that do not capture.
    A store into an aggregate which copies the object reference is a typical
    example for an operation that captures that specific value.

    For the ease of definition the `Captured` annotation can be used to annotate
    that an operation captures the value input into the annotated operand.
  }];

  let methods = [
    ExtendableInterfaceMethod<[{
      Returns whether the operation captures the value.
    }], "bool", "capturesValue", (ins "::mlir::Value":$value), [{
      bool isCaptured = false;
    }], "return isCaptured;">
  ];
}

/// Alias for 'DeclareOpInterfaceMethods<CaptureInterface>' with a more
/// descriptive name indicating that an Op does not capture anything. Prefer to
/// use this name over 'DeclareOpInterfaceMethods<CaptureInterface>' if and only
/// if 'Captured' will not be used on any operands.
defvar NoCaptures = DeclareOpInterfaceMethods<CaptureInterface>;

/// Op Operand annotation to specify that the value of the specific operand is
/// captured by the operation.
def Captured : OpVariableGen<CaptureInterface> {
  let methodGens = [
    OpVariableMethodGen<"capturesValue", [{
      isCaptured = isCaptured || $_arg.get() == value;
    }]>
  ];
}

#endif
