// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_INTERFACES_ATTR_VERIFY_INTERFACE
#define PYLIR_INTERFACES_ATTR_VERIFY_INTERFACE

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "pylir/Optimizer/Util/TablegenUtil.td"

def AttrVerifyInterface : AttrInterface<"AttrVerifyInterface"> {
  let cppNamespace = "::pylir";

  let description = [{
    This attribute interface allows verifying the structural properties of
    an attribute.
    Common use cases include attributes referencing symbols and
    checking that the referred to operation is of a specific shape and kind and
    attributes using mutable attributes and requiring the cycle, and therefore
    creation and parsing, to be complete prior to verification.
  }];

  let methods = [
    InterfaceMethod<[{
      Called to verify the structural properties of the attribute.
      `operation` is an operation that this attribute is transitively reachable
      from and can be used to emit errors using the `emitError` method.
    }], "mlir::LogicalResult", "verifyStructure",
      (ins "::mlir::Operation*":$operation,
           "::mlir::SymbolTableCollection&":$collection)>
  ];

  let extraClassDeclaration = [{
    /// Entry point to verify all attributes reachable from `operation` or any
    /// of its sub-operations.
    static mlir::LogicalResult
    verify(mlir::Operation* operation, mlir::SymbolTableCollection& table);
  }];
}

/// Trait that can be attached to any operation to automatically verify all
/// attributes reachable from it or any of its sub-operations. The mechanism
/// works by automatically implementing the `SymbolUserOpInterface` to receive
/// a `SymbolTableCollection`.
def VerifyAttributesOpTrait : OpInterfaceImplementation<SymbolUserOpInterface,
  [{
    mlir::LogicalResult verifySymbolUses(mlir::SymbolTableCollection& table) {
      return ::pylir::AttrVerifyInterface::verify(*this, table);
    }
  }]>;

#endif
