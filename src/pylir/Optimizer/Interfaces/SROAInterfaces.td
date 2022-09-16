// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_INTERFACES_SROA_INTERFACES
#define PYLIR_INTERFACES_SROA_INTERFACES

include "mlir/IR/OpBase.td"
include "pylir/Optimizer/Util/OpVariableBase.td"

def SROAAllocOpInterface : OpInterface<"SROAAllocOpInterface"> {
    let cppNamespace = "::pylir";

    let description = [{
        This interface signifies that an aggregate created by this operation should participate in SROA.
        SROA works by deconstructing an aggregate type into its elements.
        This only works if given read and write operations statically know which value within the aggregate
        they're referring to. See the `SROAReadWriteOpInterface` below for read and write operations.
    }];

    let methods = [
        InterfaceMethod<[{
            Called to check whether this operations allocation can participate in SROA. The default implementation
            simply returns success.
        }], "::mlir::LogicalResult", "canParticipateInSROA", (ins), "", [{
            return ::mlir::success();
        }]>,
        InterfaceMethod<[{
            Called by the SSABuilder in the case that a value within the aggregate created by this operation is being
            read without ever having been written to. In this case an "undefined" value of some kind should be
            materialized by this operation.

            `builder` is an `OpBuilder` with an insertion point dominating all uses of the read. `type` is the expected
            type and `loc` is the location that should be used when creating operations.
        }], "::mlir::Value", "materializeUndefined", (ins "::mlir::OpBuilder&":$builder, "::mlir::Type":$type,
                                                          "::mlir::Location":$loc)>,
        InterfaceMethod<[{
            Called during replacement of the aggregate. The purpose of this method is to be able to initialize the
            values of the aggregate. This is done via the callback `write`, which takes two arguments. The first is the
            key, which is used by all participating 'SROAReadWriteOpInterface' operations and this operation to
            identity a value within the aggregate. The second argument is the value that should be written to the value
            within the aggregate.
            If any operations should be created during the process `builder` may be used, which has its insertion point
            set to right before the operation.
        }], "void", "replaceAggregate", (ins "::mlir::OpBuilder&":$builder,
                                             "::llvm::function_ref<void(mlir::Attribute, mlir::Value)>":$write), "", [{
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
        This interface should be used on read or write operations of aggregates that should be able to participate in
        SROA. Only if all uses of an aggregate are in read or write operations implementing the interface can it be
        fully replaced.

        This interface makes use of two `OpVariable`s to ease definition. `SROAAggregate` must be attached to one
        operand which is the aggregate that is being read from or written to. When SROA checks if all uses are
        read and write ops it makes sure that these uses are the aggregate operand of read write ops.

        Additionally, one may optionally add `SROAKey` to at max one operand. The key is an operand which is used to
        find out the precise value that is being read from or written to within the aggregate. If the operation
        has such a key operand, it should be marked with `SROAKey`. This will then make sure that the key is statically
        known and pass it along as well.
    }];

    let methods = [
        ExtendableInterfaceMethod<[{
            Returns the aggregate operand which is read from or written to.
        }], "::mlir::OpOperand&", "getAggregateOperand", (ins), "", "PYLIR_UNREACHABLE;">,
        ExtendableInterfaceMethod<[{
            Returns the key operand identifying the operand that is used to identify the precise location that is read
            from or written to within the aggregate, or nullptr if no such key exists, as the operation always reads or
            writes to a fixed value within the aggregate.
        }], "::mlir::OpOperand*", "getOptionalKeyOperand", (ins), "", "return nullptr;">,
        InterfaceMethod<[{
            Called to validate the key. If `getOptionalKeyOperand` returned an operand which was then determined to
            have a static value, it may then be additionally verified by the operation to make sure the key can be used
            to precisely determine the value read from or written to within the aggregate. Should return success
            if the key is suitable.
        }], "::mlir::LogicalResult", "validateKey", (ins "::mlir::Attribute":$key), "", [{
            return mlir::success();
        }]>,
        InterfaceMethod<[{
            Called during the replacement of the aggregate. Using the callbacks `read` and `write` one can make reads
            and writes to specific values within the aggregate. These values are located via the `Attribute` passed
            as the first argument to each callbacks respectively. `read`s second argument is the type of the value
            to be read. It then returns the given value.
            `write`s second argument is the value that should be written to the value within the aggregate.
            `builder` may be used in case any operations are to be created and has its insertion point set before this
            operation. `optionalKey` is the static value of the `SROAKey` operand returned by `getOptionalKeyOperand` or
            a null attribute if no operand was annotated as such.

            If this operation has any results they're required to have no uses after this operations.
        }], "void", "replaceAggregate", (ins "::mlir::OpBuilder&":$builder, "::mlir::Attribute":$optionalKey,
                                             "::llvm::function_ref<mlir::Value(mlir::Attribute, mlir::Type)>":$read,
                                             "::llvm::function_ref<void(mlir::Attribute, mlir::Value)>":$write)>,
    ];
}

def SROAAggregate : OpVariableGen<SROAReadWriteOpInterface> {
    let methodGens = [
        OpVariableMethodGen<"getAggregateOperand", "return $_arg;">
    ];
}

def SROAKey : OpVariableGen<SROAReadWriteOpInterface> {
    let methodGens = [
        OpVariableMethodGen<"getOptionalKeyOperand", "return &$_arg;">
    ];
}

#endif
