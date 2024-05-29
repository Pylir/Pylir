// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_PY_INTERFACES_EXCEPTION_HANDLING
#define PYLIR_PY_INTERFACES_EXCEPTION_HANDLING

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "pylir/Optimizer/PylirPy/IR/PylirPyTraits.td"

def ExceptionHandlingInterface : OpInterface<"ExceptionHandlingInterface"> {
  let cppNamespace = "::pylir::Py";

  let description = [{
    Interface implemented by any operation performing CFG-base exception
    handling.
    Every such operation is a terminator that has two successors that are
    branched to if an exception was thrown or not.
    The unwind successor additionally reserves the first block argument of the
    successor block and sets it to the exception object that was thrown. It
    must be of type `!py.dynamic`.
  }];

  let methods = [
    InterfaceMethod<[{
      Returns the successor used if no exception was thrown.
    }], "::mlir::Block*", "getHappyPath", (ins)>,
    InterfaceMethod<[{
      Returns the successor used if an exception was thrown.
    }], "::mlir::Block*", "getExceptionPath", (ins)>,
    InterfaceMethod<[{
      Returns the operands passed to the block arguments of the block returned
      by `getHappyPath`.
    }], "::mlir::MutableOperandRange", "getNormalDestOperandsMutable", (ins)>,
    InterfaceMethod<[{
      Returns the operands passed to the block arguments of the block returned
      by `getExceptionPath`.
    }], "::mlir::MutableOperandRange", "getUnwindDestOperandsMutable", (ins)>,
    InterfaceMethod<[{
      Creates a non-exception handling version copy of this operation using
      `builder`.
    }],
    "::mlir::Operation*",
    "cloneWithoutExceptionHandling", (ins "::mlir::OpBuilder&":$builder)>,
  ];

  let extraSharedClassDeclaration = [{
    ::mlir::OperandRange getNormalDestOperands() {
      return static_cast<mlir::OperandRange>(
        this->getNormalDestOperandsMutable());
    }

    ::mlir::OperandRange getUnwindDestOperands() {
      return static_cast<mlir::OperandRange>(
        this->getUnwindDestOperandsMutable());
    }
  }];
}

def AddableExceptionHandlingInterface : OpInterface<"AddableExceptionHandlingInterface"> {
  let cppNamespace = "::pylir::Py";

  let description = [{
    Interface implemented by any operation for which an exception handling
    version exists. The exception handling version must be identical to the
    normal version but have the successors and extra operands required by
    `ExceptionHandlingInterface` added.

    See `CreateExceptionHandlingVariant` in `ExceptionHandlingInterface.td`
    for a macro to easily create such operations.
  }];

  let methods = [
    InterfaceMethod<[{

    }], "::mlir::Operation*", "cloneWithExceptionHandling",
      (ins "::mlir::OpBuilder&":$builder,
           "::mlir::Block*":$happy_path,
           "::mlir::Block*":$exception_path,
           "::mlir::ValueRange":$unwind_operands)>,
  ];
}

class RecFilter<Trait trait> {
  list<Trait> result = !cond(
    !isa<AddableExceptionHandling>(trait) : [],
    !isa<TraitList>(trait)
      : !foldl([]<Trait>, !cast<TraitList>(trait).traits, acc, iter,
          !listconcat(acc, RecFilter<iter>.result)),
    true : [trait]
  );
}

/// Creates a new Op as a variant of the existing Op 'op', adding the required
/// arguments, successors, traits, interface implementations and syntax
/// automatically.
/// If 'opName' is empty, its mnemonic is the mnemonic of 'op' with 'Ex' as
/// suffix.
class CreateExceptionHandlingVariant<Op op, string opName = "", bit newline = 1>
  : Op<op.opDialect, !if(!empty(opName), !strconcat(op.opName, "Ex"), opName),
    !listconcat(RecFilter<TraitList<op.traits>>.result,
      [Terminator, AttrSizedOperandSegments,
        DeclareOpInterfaceMethods<BranchOpInterface>,
        DeclareOpInterfaceMethods<ExceptionHandlingInterface>])> {

  assert !eq(!size(op.successors), 0), "Op mustn't already have successors";

  let summary = "Exception-handling variant of `" # op.opDialect.name # "."
    # op.opName # "`";
  let description = "";

  let results = op.results;
  let regions = op.regions;
  let arguments = !con(op.arguments,
            (ins Variadic<AnyType>:$normal_dest_operands,
                 Variadic<AnyType>:$unwind_dest_operands));
  let successors = (successor AnySuccessor:$happy_path,
                              AnySuccessor:$exception_path);

  let assemblyFormat = !strconcat(op.assemblyFormat,
    !if(newline, [{`\n` ` ` ` `}], " ") # [{
     `label` $happy_path ( `(` $normal_dest_operands^ `:` type($normal_dest_operands) `)`)?
           `unwind` $exception_path ( `(` $unwind_dest_operands^ `:` type($unwind_dest_operands) `)`)?
  }]);

  let extraClassDeclaration = op.extraClassDeclaration;

  let extraClassDefinition = [{

    //===------------------------------------------------------------------===//
    // BranchOpInterface implementation
    //===------------------------------------------------------------------===//

    mlir::SuccessorOperands
    $cppClass::getSuccessorOperands(unsigned index) {
      if (index == 0)
        return mlir::SuccessorOperands(getNormalDestOperandsMutable());
      return mlir::SuccessorOperands(/*exception object*/1,
              getUnwindDestOperandsMutable());
    }

    //===------------------------------------------------------------------===//
    // ExceptionHandlingInterface implementation
    //===------------------------------------------------------------------===//

    mlir::Operation*
    $cppClass::cloneWithoutExceptionHandling(::mlir::OpBuilder& builder) {
      return pylir::Py::cloneWithoutExceptionHandlingImpl(builder, *this, "}]
        # op.opDialect.name # "." # op.opName #[{");
    }
  }];
}

#endif
