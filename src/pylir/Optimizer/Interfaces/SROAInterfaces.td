// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_INTERFACES_SROA_INTERFACES
#define PYLIR_INTERFACES_SROA_INTERFACES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "pylir/Optimizer/Util/OpVariableBase.td"

def SROAAllocOpInterface : OpInterface<"SROAAllocOpInterface"> {
  let cppNamespace = "::pylir";

  let description = [{
    This interface signifies that an aggregate created by this operation should
    participate in SROA.
    SROA works by deconstructing an aggregate type into its elements.
    This only works if given read and write operations statically know which
    value within the aggregate they're referring to.
     See the `SROAReadWriteOpInterface` below for read and write operations.
  }];

  let methods = [
    InterfaceMethod<[{
      Called to check whether this operations allocation can participate in
      SROA.
      The default implementations imply returns success.
    }], "::mlir::LogicalResult", "canParticipateInSROA", (ins), "", [{
      return ::mlir::success();
    }]>,
    InterfaceMethod<[{
      Called during replacement of the aggregate.
      The purpose of this method is to be able to initialize the values of the
      aggregate.
      This is done via the callback `write`, which takes three arguments.
      The first is the key, which is used by all participating
      'SROAReadWriteOpInterface' operations and this operation to identity a
      value within the aggregate.
      The second argument is a memory resource, which acts as an additional
      scoping mechanism when later retrieving a value.
      To read a value written here, using the same key and scope is required.
      The last argument is the value that should be written to the value within
      the aggregate.
      If any operations should be created during the process `builder` may be
      used, which has its insertion point set to right before the operation.
    }], "void", "replaceAggregate", (ins
      "::mlir::OpBuilder&":$builder,
      "::llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*, mlir::Value)>":$write), "", [{
      // define nothing by default.
    }]>,
  ];

  let verify = [{
    static_assert(ConcreteOp::template hasTrait<::mlir::OpTrait::OneResult>());
    return mlir::success();
  }];
}

def SROAReadWriteOpInterface : OpInterface<"SROAReadWriteOpInterface"> {
  let cppNamespace = "::pylir";

  let description = [{
    This interface should be used on read or write operations of aggregates
    that should be able to participate in SROA.
    Only if all uses of an aggregate are in read or write operations
    implementing the interface can it be fully replaced.

    This interface makes use of two `OpVariable`s to ease definition.
    `SROAAggregate` must be attached to one operand which is the aggregate that
    is being read from or written to. When SROA checks if all uses are read and
    write ops it makes sure that these uses are the aggregate operand of read
    write ops.

    Additionally, one may optionally add `SROAKey` to at max one operand.
    The key is an operand which is used to find out the precise value that is
    being read from or written to within the aggregate.
    If the operation has such a key operand that is simply a constant operand,
    one can use `SROAKey` to autogenerate the `getKey` method.
    Otherwise, the method should override `getKey` to return an appropriate key.
  }];

  let methods = [
    ExtendableInterfaceMethod<[{
      Returns the aggregate operand which is read from or written to.
    }], "::mlir::OpOperand&", "getAggregateOperand", (ins), "", "PYLIR_UNREACHABLE;">,
    ExtendableInterfaceMethod<[{
      Returns the key identifying the operand that is used to identify the
      precise location that is read from or written to within the aggregate, or
      nullptr if no such key exists, as the operation always reads or writes to
      a fixed value within the aggregate.
      The result of this method (including nullptr) is then passed onto
      `replaceAggregate`.
      The default implementation simply returns nullptr.

      If the op cannot participate in SROA due to no suitable key being able to
      be gathered, failure should be returned.
    }], "::mlir::FailureOr<::mlir::Attribute>", "getSROAKey", (ins), "", "return mlir::Attribute{};">,
    InterfaceMethod<[{
      Called during the replacement of the aggregate.
      Using the callbacks `read` and `write` one can make reads and writes to
      specific values within the aggregate.
      These values are located via the `Attribute` passed as the first argument
      and the 'Resource' passed as the second argument to each callback
      respectively.
      `read`s third argument is the type of the value to be read.
      It then returns the given value. `write`s third argument is the value that
      should be written to the value within the aggregate.
      `builder` may be used in case any operations are to be created and has its
      insertion point set before this operation.
      `optionalKey` is the static value of the `SROAKey` operand returned by
      `getSROAKey` or a null attribute if no operand was annotated as such.

      If this operation has any results they're required to have no uses after
      this operations.
    }], "void", "replaceAggregate", (ins
      "::mlir::OpBuilder&":$builder, "::mlir::Attribute":$optionalKey,
      "::llvm::function_ref<mlir::Value(mlir::Attribute, mlir::SideEffects::Resource*, mlir::Type)>":$read,
      "::llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*, mlir::Value)>":$write)>,
  ];
}

def SROAAggregate : OpVariableGen<SROAReadWriteOpInterface> {
  let methodGens = [
    OpVariableMethodGen<"getAggregateOperand", "return $_arg;">
  ];
}

/// Common op variable annotation for ops whose 'getSROAKey' implementation
/// should simply check whether a specific key has a statically known constant
/// value.
def SROAKey : OpVariableGen<SROAReadWriteOpInterface> {
  let methodGens = [
    OpVariableMethodGen<"getSROAKey", [{
      mlir::Attribute attr;
      if (mlir::matchPattern($_arg.get(), mlir::m_Constant(&attr)))
        return attr;
      return mlir::failure();
    }]>
  ];
}

def SROAAttrInterface : AttrInterface<"SROAAttrInterface"> {
  let cppNamespace = "::pylir::Py";

  let description = [{
    Interface for attributes of aggregates to participate in SROA.
  }];

  let methods = [
    InterfaceMethod<[{
      Method to initialize the scalar values of an aggregate that is initialized
      by an attribute.
      This works similarly to the callbacks with `replaceAggregate` of
      `SROAAllocOpInterface` and `SROAReadWriteOpInterface` respectively.
      `write` is used to initialize the values within the aggregate.
      The first and second parameters denote the location within the attribute,
      and initializes it with `value`. Additionally the MLIR `type` of the value
      has to be passed as second argument.
    }], "void", "destructureAggregate",
    (ins
    "::llvm::function_ref<void(mlir::Attribute, mlir::SideEffects::Resource*, mlir::Type, mlir::Attribute)>":$write
    )>
  ];
}

#endif
