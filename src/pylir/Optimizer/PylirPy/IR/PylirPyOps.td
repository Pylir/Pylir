// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_PY_OPS_TABLEGEN
#define PYLIR_PY_OPS_TABLEGEN

include "pylir/Optimizer/PylirPy/IR/PylirPyAttributes.td"
include "pylir/Optimizer/PylirPy/IR/PylirPyBase.td"
include "pylir/Optimizer/PylirPy/IR/PylirPyEnums.td"
include "pylir/Optimizer/PylirPy/IR/PylirPyTraits.td"
include "pylir/Optimizer/PylirPy/IR/PylirPyTypes.td"
include "pylir/Optimizer/PylirPy/Interfaces/ExceptionHandlingInterface.td"
include "pylir/Optimizer/PylirPy/Interfaces/CopyObjectInterface.td"
include "pylir/Optimizer/PylirPy/Interfaces/OnlyReadsValueInterface.td"
include "pylir/Optimizer/Interfaces/AttrVerifyInterface.td"
include "pylir/Optimizer/Interfaces/MemoryFoldInterface.td"
include "pylir/Optimizer/Interfaces/CaptureInterface.td"
include "pylir/Optimizer/Interfaces/SROAInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Helper Classes
//===----------------------------------------------------------------------===//

/// Common base class used to define an op in the PylirPy dialect.
class PylirPy_Op<string mneomic, list<Trait> traits = []>
  : Op<PylirPy_Dialect, mneomic, traits>;

/// Resource declarations in ODS. See PylirPyOps.hpp for definition.
def GlobalResource : Resource<"::pylir::Py::GlobalResource">;
def ObjectResource : Resource<"::pylir::Py::ObjectResource">;
def ListResource : Resource<"::pylir::Py::ListResource">;
def DictResource : Resource<"::pylir::Py::DictResource">;

//===----------------------------------------------------------------------===//
// Basic Ops
//===----------------------------------------------------------------------===//

def PylirPy_ConstantOp : PylirPy_Op<"constant", [ConstantLike, NoMemoryEffect,
  KnownType<"Tuple">]> {
  let arguments = (ins AnyAttrOf<[PylirPy_UnboundAttr, ObjectAttrInterface,
    PylirPy_GlobalValueAttr]>:$constant);
  let results = (outs DynamicType:$result);

  let assemblyFormat = [{
    `(` $constant `)` attr-dict
  }];

  let hasFolder = 1;

  let description = [{
    Creates a constant value from the given attribute.
    Writing to a value returned from this op is undefined behaviour.
  }];
}

def PylirPy_TypeOfOp : PylirPy_Op<"typeOf", [NoMemoryEffect, AlwaysBound,
  NoCaptures]> {
  let arguments = (ins DynamicType:$object);
  let results = (outs DynamicType:$result);

  let assemblyFormat = "$object attr-dict";

  let description = [{
    Retrieves the type of the object. This will always be an instance of
    `builtins.type` or a subclass.
  }];

  let hasFolder = 1;
}

def PylirPy_GetSlotOp : PylirPy_Op<"getSlot", [NoCaptures,
  DeclareOpInterfaceMethods<MemoryFoldInterface>,
  DeclareOpInterfaceMethods<SROAReadWriteOpInterface>]> {

  let arguments = (ins
    Arg<DynamicType, "", [MemRead<ObjectResource>, SROAAggregate]>:$object,
    Arg<Index, "", [SROAKey]>:$slot
  );

  let results = (outs DynamicType:$result);

  let assemblyFormat = "$object `[` $slot `]` attr-dict";

  let description = [{
    Retrieves the value stored in the slot with index `$slot` inside of
    `$object`.
    If `$slot` is greater or equal to the amount of slots of `$object` it is
    undefined behaviour.
    `$result` will contain the value in the slot, or unbound value if it was
    never assigned to.
  }];

  let hasFolder = 1;
}

def PylirPy_SetSlotOp : PylirPy_Op<"setSlot", [
  DeclareOpInterfaceMethods<CaptureInterface>,
  DeclareOpInterfaceMethods<SROAReadWriteOpInterface>]> {

  let arguments = (ins
    Arg<DynamicType, "", [MemWrite<ObjectResource>, SROAAggregate]>:$object,
    Arg<Index, "", [SROAKey]>:$slot,
    Arg<DynamicType, "", [Captured]>:$value);
  let results = (outs);

  let assemblyFormat = "$object `[`$slot`]` `to` $value attr-dict";

  let description = [{
    Sets the value of the slot with index `$slot` inside of `$object` to
    `$value`.
    If `$slot` is greater or equal to the amount of slots of `$object` it is
    undefined behaviour.
  }];
}

def PylirPy_IsUnboundValueOp : PylirPy_Op<"isUnboundValue", [NoMemoryEffect,
  NoCaptures]> {
  let summary = "checks whether the value is an unbound value";

  let arguments = (ins DynamicType:$value);
  let results = (outs I1:$result);

  let assemblyFormat = "$value attr-dict";

  let hasFolder = 1;
}

def PylirPy_IsOp : PylirPy_Op<"is", [NoMemoryEffect, Commutative]> {
  let arguments = (ins DynamicType:$lhs, DynamicType:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Dictionary Ops
//===----------------------------------------------------------------------===//

def PylirPy_DictTryGetItemOp : PylirPy_Op<"dict_tryGetItem", [NoCaptures,
  DeclareOpInterfaceMethods<MemoryFoldInterface>,
  DeclareOpInterfaceMethods<SROAReadWriteOpInterface, ["getSROAKey"]>]> {
  let arguments = (ins
    Arg<DynamicType, "", [MemRead<DictResource>, SROAAggregate]>:$dict,
    Arg<DynamicType, "", [SROAKey]>:$key, Index:$hash
  );

  let results = (outs DynamicType:$result);

  let assemblyFormat = [{
    $dict `[` $key `hash` `(` $hash `)` `]` attr-dict
  }];

  let description = [{
    This op looks up the value of `$key` with the help of `$hash` in the
    dictionary `$dict`.
    If found it'll be returned in `$result`.
    If not found, `$result` is an `unbound` value.
    If `$dict` is not really a dict (or a subclass of) the behaviour is
    undefined.
  }];

  let hasFolder = 1;
}

def PylirPy_DictSetItemOp : PylirPy_Op<"dict_setItem", [
  DeclareOpInterfaceMethods<SROAReadWriteOpInterface, ["getSROAKey"]>,
  DeclareOpInterfaceMethods<CaptureInterface>]> {

  let arguments = (ins
    Arg<DynamicType, "", [MemWrite<DictResource>, SROAAggregate]>:$dict,
    Arg<DynamicType, "", [SROAKey, Captured]>:$key, Index:$hash,
    Arg<DynamicType, "", [Captured]>:$value);

  let results = (outs);

  let assemblyFormat = [{
    $dict `[` $key `hash` `(` $hash `)` `]` `to` $value attr-dict
  }];

  let description = [{
    This op inserts the value `$value` into the dictionary with the key `$key`
    with the help of `$hash`.
    If `$key` already exists the previous value gets overwritten.
    If `$dict` is not a dictionary (or a subclass of) the behaviour is
    undefined.
  }];
}

def PylirPy_DictDelItemOp : PylirPy_Op<"dict_delItem", [
  DeclareOpInterfaceMethods<CaptureInterface>,
  DeclareOpInterfaceMethods<SROAReadWriteOpInterface, ["getSROAKey"]>]> {

  let arguments = (ins
    Arg<DynamicType, "", [MemWrite<DictResource>, SROAAggregate]>:$dict,
    Arg<DynamicType, "", [Captured, SROAKey]>:$key,
    Index:$hash);

  let results = (outs I1:$existed);

  let assemblyFormat = "$key `hash` `(` $hash `)` `from` $dict attr-dict";

  let description = [{
    This op deletes the entry identified by `$key` with the help of `$hash` from
    the dictionary.
    If the dictionary did not contain an entry with the given key the result is
    0, otherwise 1.
    If `$dict` is not a dictionary (or a subclass of) the behaviour is
    undefined.
  }];
}

def PylirPy_DictLenOp : PylirPy_Op<"dict_len", [NoCaptures,
  DeclareOpInterfaceMethods<MemoryFoldInterface>,
  DeclareOpInterfaceMethods<SROAReadWriteOpInterface>]> {
  let arguments = (ins
    Arg<DynamicType, "", [MemRead<DictResource>, SROAAggregate]>:$input
  );
  let results = (outs Index:$result);

  let assemblyFormat = [{
    $input attr-dict
  }];

  let description = [{
    Returns the amount of key-value pairs stored in `$input`.
  }];

  let hasFolder = 1;
}

def PylirPy_MakeDictOp : PylirPy_Op<"makeDict", [AttrSizedOperandSegments,
  AlwaysBound, AddableExceptionHandling<"MakeDictExOp">, KnownType<"Dict">,
  DeclareOpInterfaceMethods<SROAAllocOpInterface,
    ["canParticipateInSROA", "replaceAggregate"]>]> {
  let arguments = (ins
            Variadic<DynamicType>:$keys,
            Variadic<Index>:$hashes,
            Variadic<DynamicType>:$values,
            DenseI32ArrayAttr:$mapping_expansion
            );

  let results = (outs Res<DynamicType, "",
    [MemAlloc, MemWrite<ObjectResource>, MemWrite<DictResource>]>:$result);

  let builders = [
      OpBuilder<(ins CArg<"llvm::ArrayRef<::pylir::Py::DictArg>","{}">:$args)>
  ];

  let assemblyFormat = [{
      custom<MappingArguments>($keys, $hashes, $values, $mapping_expansion)
      attr-dict
  }];

  let extraClassDeclaration = [{
    llvm::iterator_range<::pylir::Py::DictArgsIterator> getDictArgs() {
      return llvm::make_range(DictArgsIterator::begin(*this),
                              DictArgsIterator::end(*this));
    }

    struct BuilderArgs
    {
      llvm::SmallVector<mlir::Value> keys;
      llvm::SmallVector<mlir::Value> hashes;
      llvm::SmallVector<mlir::Value> values;
      llvm::SmallVector<std::int32_t> mappingExpansion;
    };

    template <class R>
    static BuilderArgs deconstructBuilderArg(R args)
    {
      llvm::SmallVector<mlir::Value> keys;
      llvm::SmallVector<mlir::Value> hashes;
      llvm::SmallVector<mlir::Value> values;
      llvm::SmallVector<std::int32_t> mappingExpansion;
      for (const auto& iter : llvm::enumerate(args))
      {
        pylir::match(
          iter.value(),
          [&](const DictEntry& entry)
          {
            keys.push_back(entry.key);
            hashes.push_back(entry.hash);
            values.push_back(entry.value);
          },
          [&](Py::MappingExpansion expansion)
          {
            keys.push_back(expansion.value);
            mappingExpansion.push_back(iter.index());
          });
      }
      return {std::move(keys), std::move(hashes), std::move(values),
              std::move(mappingExpansion)};
    }
  }];

  let hasFolder = 1;
}

def PylirPy_MakeDictExOp : CreateExceptionHandlingVariant<PylirPy_MakeDictOp> {
  let builders = [
    OpBuilder<(ins "llvm::ArrayRef<::pylir::Py::DictArg>":$keyValues,
             "mlir::Block*":$happyPath,
             "mlir::ValueRange":$normalDestOperands,
             "mlir::Block*":$unwindPath,
             "mlir::ValueRange":$unwindDestOperands
     )>
  ];

  let hasFolder = 1;

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Tuple Ops
//===----------------------------------------------------------------------===//

def PylirPy_TupleGetItemOp : PylirPy_Op<"tuple_getItem", [NoMemoryEffect,
  AlwaysBound, NoCaptures,
  DeclareOpInterfaceMethods<OnlyReadsValueInterface>]> {
  let arguments = (ins
    Arg<DynamicType, "", [OnlyReadsValue]>:$tuple,
    Index:$index);
  let results = (outs DynamicType:$result);

  let assemblyFormat = [{
    $tuple `[` $index `]` attr-dict
  }];

  let description = [{
    This op fetches the object stored at `$index` in `$tuple`.
    If `$tuple` is not really a tuple (or subclass of) or `$index` is out of
    range for the tuple, the behaviour is undefined.
  }];

  let hasFolder = 1;
}

def PylirPy_TupleLenOp : PylirPy_Op<"tuple_len", [NoMemoryEffect, NoCaptures,
  DeclareOpInterfaceMethods<OnlyReadsValueInterface>]> {
  let arguments = (ins Arg<DynamicType, "", [OnlyReadsValue]>:$input);
  let results = (outs Index:$result);

  let assemblyFormat = [{
    $input attr-dict
  }];

  let description = [{
    This op returns the length of `$input` as the given integer type.
    If `$tuple` is not really a tuple (or a subclass of) then the behaviour of
    is undefined.
  }];

  let hasFolder = 1;
}

// TODO: I don't think this should exist but don't have better option until
// iterators work
def PylirPy_TuplePrependOp : PylirPy_Op<"tuple_prepend", [NoMemoryEffect,
  AlwaysBound, ReturnsImmutable, KnownType<"Tuple">,
  DeclareOpInterfaceMethods<OnlyReadsValueInterface>,
  DeclareOpInterfaceMethods<CaptureInterface>]> {
  let arguments = (ins
    Arg<DynamicType, "", [Captured]>:$input,
    Arg<DynamicType, "", [OnlyReadsValue]>:$tuple);
  let results = (outs DynamicType:$result);

  let assemblyFormat = [{
    $input `,` $tuple attr-dict
  }];

  let description = [{

  }];

  let hasFolder = 1;
}

def PylirPy_TupleDropFrontOp : PylirPy_Op<"tuple_dropFront", [NoMemoryEffect,
  AlwaysBound, ReturnsImmutable, KnownType<"Tuple">, NoCaptures,
  DeclareOpInterfaceMethods<OnlyReadsValueInterface>]> {

  let arguments = (ins
    Index:$count,
    Arg<DynamicType, "", [OnlyReadsValue]>:$tuple);
  let results = (outs DynamicType:$result);

  let assemblyFormat = [{
    $count `,` $tuple attr-dict
  }];

  let hasFolder = 1;
}

def PylirPy_TupleCopyOp : PylirPy_Op<"tuple_copy", [AlwaysBound,
  KnownTypeObjectInterface, DeclareOpInterfaceMethods<CaptureInterface>,
  DeclareOpInterfaceMethods<CopyObjectInterface>,
  DeclareOpInterfaceMethods<OnlyReadsValueInterface>,
  DeclareOpInterfaceMethods<SROAAllocOpInterface>]> {

  let arguments = (ins Arg<DynamicType, "", [Copied, OnlyReadsValue]>:$tuple,
             Arg<DynamicType, "", [Captured]>:$type_object);

  let results = (outs Res<DynamicType, "", [MemAlloc,
                      MemWrite<ObjectResource>]>:$result);

  let description = [{
    Creates a new tuple object of type `$type_object` and initializes it by
    copying the tuple parts of `$tuple`.
    This results in a copy of all references within `$tuple` as well as the
    size. No copy of any slots are performed.

    If `$tuple` is not a tuple or a subclass of the behaviour is undefined.
    If `$type_object` is not the tuple type object or a type object deriving
    from tuple the behaviour is undefined.
  }];

  let assemblyFormat = [{
    $tuple `:` $type_object attr-dict
  }];

  let hasFolder = 1;
}

def PylirPy_TupleContainsOp : PylirPy_Op<"tuple_contains", [NoMemoryEffect,
  NoCaptures, DeclareOpInterfaceMethods<OnlyReadsValueInterface>]> {

  let arguments = (ins
    Arg<DynamicType, "", [OnlyReadsValue]>:$tuple,
    DynamicType:$element);
  let results = (outs I1:$success);

  let assemblyFormat = "$element `in` $tuple attr-dict";

  let description = [{
    This op does a linear search through the given tuple, checking whether
    `$element` is contained within.
    Equality is determined via identity comparison.
    If it is `$success` returns `true`, `false` otherwise.
    This is used to check whether a type is a subclass of another.
    If `$tuple` is not really a tuple (or subclass of) the behaviour is
    undefined.
  }];

  let hasFolder = 1;
}

def PylirPy_MakeTupleOp : PylirPy_Op<"makeTuple", [AlwaysBound,
  KnownType<"Tuple">, ReturnsImmutable,
  AddableExceptionHandling<"MakeTupleExOp">,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins Variadic<DynamicType>:$arguments,
             DenseI32ArrayAttr:$iter_expansion
             );

  let results = (outs DynamicType:$result);

  let assemblyFormat = [{
    custom<IterArguments>($arguments, $iter_expansion) attr-dict
  }];

  let builders = [
    OpBuilder<(ins CArg<"llvm::ArrayRef<::pylir::Py::IterArg>","{}">:$args)>
  ];

  let hasFolder = 1;

  let description = [{
    This op creates a tuple from the given arguments.
    If an argument is prefixed by a `*` then it must be iterable and its
    elements will be copied into the tuple at the given position.
    As this operation creates a tuple it can never alias another tuple created
    by this op unless both are turned into the same constants
    (at which point aliasing becomes irrelevant).

    This op as well as the constant op are the only ones capable of creating a
    tuple.
    After creation it is impossible to be modified.
    If it contains objects that are mutable however, those may still be
    modified.
    This way one is capable of arguing about the tuple in a typical SSA fashion.
  }];

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    llvm::SmallVector<::pylir::Py::IterArg> getIterArgs();
  }];
}

def PylirPy_MakeTupleExOp
  : CreateExceptionHandlingVariant<PylirPy_MakeTupleOp> {

  let builders = [
    OpBuilder<(ins "llvm::ArrayRef<::pylir::Py::IterArg>":$args,
             "mlir::Block*":$happyPath,
             "mlir::ValueRange":$normalDestOperands,
             "mlir::Block*":$unwindPath,
             "mlir::ValueRange":$unwindDestOperands
     )>
  ];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// List Ops
//===----------------------------------------------------------------------===//

def PylirPy_ListGetItemOp : PylirPy_Op<"list_getItem", [AlwaysBound, NoCaptures,
  DeclareOpInterfaceMethods<SROAReadWriteOpInterface>]> {

  let arguments = (ins
    Arg<DynamicType, "", [MemRead<ListResource>, SROAAggregate]>:$list,
    Arg<Index, "", [SROAKey]>:$index);

  let results = (outs DynamicType:$result);

  let assemblyFormat = [{
    $list `[` $index `]` attr-dict
  }];

  let description = [{
    This op gets the item referred to by the `$index` inside of the `$list`.
    If `$list` is not really a list (or subclass of) the behaviour is undefined.
    If `$index` is greater or equal to the length of the list the behaviour is
    undefined.
  }];
}

def PylirPy_ListSetItemOp : PylirPy_Op<"list_setItem", [
  DeclareOpInterfaceMethods<CaptureInterface>,
  DeclareOpInterfaceMethods<SROAReadWriteOpInterface>]> {

  let arguments = (ins
    Arg<DynamicType, "",[MemWrite<ListResource>, SROAAggregate]>:$list,
    Arg<Index, "", [SROAKey]>:$index,
    Arg<DynamicType, "", [Captured]>:$element);

  let results = (outs);

  let assemblyFormat = [{
    $list `[` $index `]` `to` $element attr-dict
  }];

  let description = [{
    This op sets the item referred to by the `$index` inside of the
    `$list` to `$element`.
    If `$list` is not really a list (or subclass of) the behaviour is undefined.
    If `$index` is greater or equal to the length of the list the behaviour is
    undefined.
  }];
}

def PylirPy_ListLenOp : PylirPy_Op<"list_len", [NoCaptures,
  DeclareOpInterfaceMethods<MemoryFoldInterface>,
  DeclareOpInterfaceMethods<SROAReadWriteOpInterface>]> {

  let arguments = (ins
    Arg<DynamicType, "", [MemRead<ListResource>, SROAAggregate]>:$list);
  let results = (outs Index:$result);

  let assemblyFormat = [{
    $list attr-dict
  }];

  let description = [{
    This op returns the length of `$input`.
    If `$input` is not really a list (or a subclass of) then the behaviour is
    undefined.
  }];
}

def PylirPy_ListResizeOp : PylirPy_Op<"list_resize", [
  DeclareOpInterfaceMethods<SROAReadWriteOpInterface>, NoCaptures]> {
  let arguments = (ins
    Arg<DynamicType, "", [MemWrite<ListResource>, SROAAggregate]>:$list,
    Index:$length);
  let results = (outs);

  let assemblyFormat = [{
    $list `to` $length attr-dict
  }];

  let description = [{
    This op resizes the `$list` to `$length` amount of elements, allowing both
    growing and shrinking.
    If this op caused the `$list` to grow then the newly added accessible
    elements are initially undefined and reading them prior to setting them is
    undefined behaviour.
    If `$list` is not really a list (or subclass of) the behaviour is undefined.
  }];
}

def PylirPy_ListToTupleOp : PylirPy_Op<"list_toTuple", [AlwaysBound,
  KnownType<"Tuple">, ReturnsImmutable, NoCaptures]> {
  let arguments = (ins Arg<DynamicType, "", [MemRead<ListResource>]>:$list);
  let results = (outs DynamicType:$result);

  let assemblyFormat = [{
    $list attr-dict
  }];

  let description = [{
    This op creates a tuple from the elements in `$list`. The list remains
    unmodified.
    If `$list` is not really a list (or subclass of) the behaviour is undefined.
  }];
}

def PylirPy_MakeListOp : PylirPy_Op<"makeList", [AlwaysBound,
  AddableExceptionHandling<"MakeListExOp">, KnownType<"List">,
  DeclareOpInterfaceMethods<SROAAllocOpInterface, ["canParticipateInSROA",
   "replaceAggregate"]>]> {
  let arguments = (ins Variadic<DynamicType>:$arguments,
             DenseI32ArrayAttr:$iter_expansion
             );
  let results = (outs
    Res<DynamicType, "", [MemAlloc,
      MemWrite<ObjectResource>,
      MemWrite<ListResource>]>:$result);

  let assemblyFormat = [{
    custom<IterArguments>($arguments, $iter_expansion) attr-dict
  }];

  let builders = [
    OpBuilder<(ins CArg<"llvm::ArrayRef<::pylir::Py::IterArg>","{}">:$args)>
  ];

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    llvm::SmallVector<::pylir::Py::IterArg> getIterArgs();
  }];
}

def PylirPy_MakeListExOp : CreateExceptionHandlingVariant<PylirPy_MakeListOp> {
  let builders = [
    OpBuilder<(ins "llvm::ArrayRef<::pylir::Py::IterArg>":$args,
             "mlir::Block*":$happyPath,
             "mlir::ValueRange":$normalDestOperands,
             "mlir::Block*":$unwindPath,
             "mlir::ValueRange":$unwindDestOperands
     )>
  ];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Set Ops
//===----------------------------------------------------------------------===//

def PylirPy_MakeSetOp : PylirPy_Op<"makeSet", [AlwaysBound,
  AddableExceptionHandling<"MakeSetExOp">, KnownType<"Set">]> {
  let arguments = (ins Variadic<DynamicType>:$arguments,
             DenseI32ArrayAttr:$iter_expansion
             );

  let results = (outs
    Res<DynamicType, "", [MemAlloc, MemWrite<ObjectResource>]>:$result);

  let builders = [
    OpBuilder<(ins CArg<"llvm::ArrayRef<::pylir::Py::IterArg>","{}">:$args)>
  ];

  let assemblyFormat = [{
    custom<IterArguments>($arguments, $iter_expansion) attr-dict
  }];

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    llvm::SmallVector<::pylir::Py::IterArg> getIterArgs();
  }];
}

def PylirPy_MakeSetExOp : CreateExceptionHandlingVariant<PylirPy_MakeSetOp> {
  let builders = [
    OpBuilder<(ins "llvm::ArrayRef<::pylir::Py::IterArg>":$args,
             "mlir::Block*":$happyPath,
             "mlir::ValueRange":$normalDestOperands,
             "mlir::Block*":$unwindPath,
             "mlir::ValueRange":$unwindDestOperands
     )>
  ];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Function Ops
//===----------------------------------------------------------------------===//

def PylirPy_FunctionCallOp : PylirPy_Op<"function_call", [AlwaysBound,
  AddableExceptionHandling<"FunctionInvokeOp", ["Single", "Variadic"]>,
  DeclareOpInterfaceMethods<CallOpInterface>]> {
  let arguments = (ins DynamicType:$function,
                       Variadic<DynamicType>:$call_operands);
  let results = (outs DynamicType:$result);

  let assemblyFormat = [{
    $function `(` $call_operands `)` attr-dict
  }];

  let hasCanonicalizeMethod = 1;
}

def PylirPy_FunctionInvokeOp
  : CreateExceptionHandlingVariant<PylirPy_FunctionCallOp, "function_invoke"> {
  let summary = "invoke operation";

  let hasCanonicalizeMethod = 1;
}

def PylirPy_MakeFuncOp : PylirPy_Op<"makeFunc", [AlwaysBound,
  KnownType<"Function">, DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<SROAAllocOpInterface>]> {
  let arguments = (ins FlatSymbolRefAttr:$function,
                       Variadic<AnyType>:$closure_args);

  let description = [{
    Creates a new instance of a python function that refers to `function`.
    `closure_args` are extra operands that are stored as part of the function
    object and can only be read using `py.function_closureArg`, where the index
    refers to the position within `closure_args`.
  }];

  let results = (outs
    Res<DynamicType, "", [MemAlloc, MemWrite<ObjectResource>]>:$result);

  let assemblyFormat = [{
    $function (`[` $closure_args^ `:` type($closure_args) `]`)? attr-dict
  }];
}

def PylirPy_FunctionGetClosureArgOp : PylirPy_Op<"function_closureArg",
  [NoMemoryEffect, InferTypeOpAdaptor]> {
  let arguments = (ins DynamicType:$function,
                       I32Attr:$index,
                       TypeArrayAttr:$closure_types);

  let results = (outs AnyType:$result);

  let description = [{
    Returns the closure argument at the given index within `function`.
    `function` must have been created with closure arguments whose types match
    `closure_types` or whose type of closure arguments match up until and
    including `index`.
  }];

  let assemblyFormat = [{
    $function `[` $index `]` `:` $closure_types attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Object Ops
//===----------------------------------------------------------------------===//

def PylirPy_ObjectHashOp : PylirPy_Op<"object_hash", [NoMemoryEffect,
  NoCaptures]> {
  let arguments = (ins DynamicType:$object);
  let results = (outs Index:$hash);

  let assemblyFormat = "$object attr-dict";

  let description = [{
    Returns a hash value from an arbitrary object using its identity.
  }];
}

def PylirPy_ObjectIdOp : PylirPy_Op<"object_id", [NoMemoryEffect, NoCaptures]> {
  let arguments = (ins DynamicType:$object);
  let results = (outs Index:$id);

  let assemblyFormat = "$object attr-dict";
}

def PylirPy_MakeObjectOp : PylirPy_Op<"makeObject", [AlwaysBound,
  KnownTypeObjectInterface, DeclareOpInterfaceMethods<SROAAllocOpInterface>]> {
  let arguments = (ins DynamicType:$type_object);

  let results = (outs
    Res<DynamicType, "", [MemAlloc, MemWrite<ObjectResource>]>:$result);

  let description = [{
    Creates a new object of the type of the given type object.
    If `$type_object` is not an instance of `builtins.type` the behaviour is
    undefined.
  }];

  let assemblyFormat = "$type_object attr-dict";
}

//===----------------------------------------------------------------------===//
// Type Ops
//===----------------------------------------------------------------------===//

def PylirPy_MakeTypeOp : PylirPy_Op<"makeType", [AlwaysBound,
  KnownType<"Type">,
]> {

  let description = [{
    Creates a new type object with the given name, MRO tuple and slots tuple.
    The MRO tuple must be a tuple of type objects.
    It is not captured, but rather copied with the result of the `makeType`
    operation prepended.
    The slots tuple must be a tuple of strings and is captured.
  }];

  let arguments = (ins
    DynamicType:$name,
    DynamicType:$mro_tuple,
    DynamicType:$slots_tuple
  );

  let results = (outs
    Res<DynamicType, "", [MemAlloc,
      MemWrite<ObjectResource>,
      MemWrite<ListResource>]>:$result);

  let assemblyFormat = [{
    `(` `name` `=` $name `,`
        `mro` `=` $mro_tuple `,`
        `slots` `=` $slots_tuple `)` attr-dict
  }];
}

def PylirPy_TypeMROOp : PylirPy_Op<"type_mro", [NoMemoryEffect, AlwaysBound,
  ReturnsImmutable, KnownType<"Tuple">, NoCaptures]> {
  let arguments = (ins DynamicType:$type_object);
  let results = (outs DynamicType:$result);

  let assemblyFormat = "$type_object attr-dict";

  let hasFolder = 1;
}

def PylirPy_TypeSlotsOp : PylirPy_Op<"type_slots", [NoMemoryEffect, AlwaysBound,
  ReturnsImmutable, KnownType<"Tuple">, NoCaptures]> {
  let arguments = (ins DynamicType:$type_object);
  let results = (outs DynamicType:$result);

  let description = [{
    Returns the tuple containing the names of all slots of an instance of
    `$type_object` as strings.
    If `$type_object` is not really a type (or a subclass of) the behaviour is
    undefined.
  }];

  let assemblyFormat = "$type_object attr-dict";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// String Ops
//===----------------------------------------------------------------------===//

def PylirPy_StrCopyOp : PylirPy_Op<"str_copy", [KnownTypeObjectInterface,
  DeclareOpInterfaceMethods<CaptureInterface>, AlwaysBound,
  DeclareOpInterfaceMethods<CopyObjectInterface>,
  DeclareOpInterfaceMethods<OnlyReadsValueInterface>,
  DeclareOpInterfaceMethods<SROAAllocOpInterface>]> {

  let arguments = (ins Arg<DynamicType, "", [Copied, OnlyReadsValue]>:$string,
             Arg<DynamicType, "", [Captured]>:$type_object);

  let results = (outs Arg<DynamicType, "", [MemAlloc,
    MemWrite<ObjectResource>]>:$result);

  let assemblyFormat = "$string `:` $type_object attr-dict";

  let description = [{
    Creates a copy of `$string` which will have the type `$type_object`.
    If `$string` is not really a string (or subclass of) or `$type_object` is
    not `builtins.str` (or subclass of) the behaviour is undefined.
  }];
}

def PylirPy_StrConcatOp : PylirPy_Op<"str_concat", [NoMemoryEffect, AlwaysBound,
 KnownType<"Str">, ReturnsImmutable, NoCaptures,
 DeclareOpInterfaceMethods<OnlyReadsValueInterface>]> {

  let arguments = (ins
    Arg<Variadic<DynamicType>, "", [OnlyReadsValue]>:$strings);

  let results = (outs DynamicType:$result);

  let assemblyFormat = "$strings attr-dict";

  let description = [{
    Creates a new string by concatenating all `$strings` from left to right.
    If any of the operands are not really a string (or a subclass of) the
    behaviour is undefined.
  }];

  let hasFolder = 1;
}

def PylirPy_StrHashOp : PylirPy_Op<"str_hash", [NoMemoryEffect, NoCaptures,
  DeclareOpInterfaceMethods<OnlyReadsValueInterface>]> {

  let arguments = (ins Arg<DynamicType, "", [OnlyReadsValue]>:$object);

  let results = (outs Index:$hash);

  let assemblyFormat = "$object attr-dict";

  let description = [{
    Returns the hash value for the string `$object`.
    If `$object` is not really a string (or a subclass of) the behaviour is
    undefined.
  }];
}

def PylirPy_StrEqualOp : PylirPy_Op<"str_equal", [NoMemoryEffect, NoCaptures,
  DeclareOpInterfaceMethods<OnlyReadsValueInterface>]> {

  let arguments = (ins
    Arg<DynamicType, "", [OnlyReadsValue]>:$lhs,
    Arg<DynamicType, "", [OnlyReadsValue]>:$rhs);

  let results = (outs I1:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let description = [{
    Checks whether two strings are equal.
    If `$lhs` or `$rhs` are not really strings (or a subclass of) the behaviour
    is undefined.
  }];
}

//===----------------------------------------------------------------------===//
// Int Ops
//===----------------------------------------------------------------------===//

def PylirPy_IntFromSignedOp : PylirPy_Op<"int_fromSigned",
  [AlwaysBound, KnownType<"Int">, ReturnsImmutable, NoMemoryEffect]> {

  let arguments = (ins Index:$input);
  let results = (outs DynamicType:$result);

  let assemblyFormat = "$input attr-dict";

  let description = [{
    Interprets an index as signed and converts it into a python integer.
  }];

  let hasFolder = 1;
}

def PylirPy_IntFromUnsignedOp : PylirPy_Op<"int_fromUnsigned",
  [AlwaysBound, KnownType<"Int">, ReturnsImmutable, NoMemoryEffect]> {

  let arguments = (ins Index:$input);
  let results = (outs DynamicType:$result);

  let assemblyFormat = "$input attr-dict";

  let description = [{
    Interprets an index as unsigned and converts it into a python integer.
  }];

  let hasFolder = 1;
}

def PylirPy_IntToIndexOp : PylirPy_Op<"int_toIndex", [NoMemoryEffect,
  NoCaptures]> {
  let arguments = (ins DynamicType:$input);
  let results = (outs Index:$result);

  let assemblyFormat = "$input attr-dict";

  let description = [{
    Converts the python integer `$input` into an index.
    The python integers value is interpreted as if encoded via twos complement
    with the bit-width of the index type.
    That means any python integer created via `py.fromUnsigned` or
    `py.int_fromSigned` can be converted back to the exact index value it was
    created with.

    If `$input` is outside the range of values representable by `index`, that is
    less than $-2^{\text{index_width}-1}$ or larger than
    $2^{\text{index_width}}-1$, the behaviour is undefined.
    If `$input` is not really an int (or subclass of) the behaviour is
    undefined.
  }];

  let hasFolder = 1;
}

def PylirPy_IntCmpOp : PylirPy_Op<"int_cmp", [NoMemoryEffect, NoCaptures]> {
  let arguments = (ins
    PylirPy_IntCmpKindAttr:$pred, DynamicType:$lhs, DynamicType:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = "$pred $lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def PylirPy_IntAddOp : PylirPy_Op<"int_add", [NoCaptures, Commutative,
  AlwaysBound, NoMemoryEffect, KnownType<"Int">, ReturnsImmutable]> {
  let arguments = (ins DynamicType:$lhs, DynamicType:$rhs);
  let results = (outs DynamicType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

def PylirPy_IntToStrOp : PylirPy_Op<"int_toStr", [NoCaptures, AlwaysBound,
                          NoMemoryEffect, KnownType<"Str">, ReturnsImmutable]> {
  let arguments = (ins DynamicType:$input);
  let results = (outs DynamicType:$result);

  let assemblyFormat = "$input attr-dict";

  let description = [{
    Converts a python integer into a python string.
    If `$input` is not really an int (or subclass of) the behaviour is
    undefined.
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Bool Ops
//===----------------------------------------------------------------------===//

def PylirPy_BoolToI1Op : PylirPy_Op<"bool_toI1", [NoMemoryEffect, NoCaptures]> {
  let arguments = (ins DynamicType:$input);
  let results = (outs I1:$result);

  let assemblyFormat = "$input attr-dict";

  let hasFolder = 1;

  let description = [{
    This op converts the boolean given by `$input` into an `I1` value that is
    usable for eg. conditional branches.
    If `$input` is not really a `bool` (or a subclasses of) the behaviour is
    undefined.
  }];
}

def PylirPy_BoolFromI1Op : PylirPy_Op<"bool_fromI1", [AlwaysBound,
  KnownType<"Bool">, ReturnsImmutable, NoMemoryEffect, NoCaptures]> {

  let arguments = (ins I1:$input);
  let results = (outs DynamicType:$result);

  let assemblyFormat = "$input attr-dict";

  let hasFolder = 1;

  let description = [{
    This op converts an `I1` given by `$input` into a python bool object_
  }];
}

//===----------------------------------------------------------------------===//
// Float Ops
//===----------------------------------------------------------------------===//

def PylirPy_FloatFromF64 : PylirPy_Op<"float_fromF64", [AlwaysBound,
  KnownType<"Float">, ReturnsImmutable, NoMemoryEffect, NoCaptures]> {

  let arguments = (ins F64:$input);
  let results = (outs DynamicType:$result);

  let assemblyFormat = "$input attr-dict";
}

def PylirPy_FloatToF64 : PylirPy_Op<"float_toF64", [NoCaptures, NoMemoryEffect,
  DeclareOpInterfaceMethods<OnlyReadsValueInterface>]> {

  let arguments = (ins Arg<DynamicType, "", [OnlyReadsValue]>:$input);
  let results = (outs F64:$result);

  let assemblyFormat = "$input attr-dict";
}

//===----------------------------------------------------------------------===//
// Miscellaneous Ops
//===----------------------------------------------------------------------===//

def PylirPy_PrintOp : PylirPy_Op<"intr_print", [NoCaptures,
  MemoryEffects<[MemWrite]>]> {
  let arguments = (ins DynamicType:$string);
  let results = (outs);

  let assemblyFormat = "$string attr-dict";
}

def PylirPy_MROLookupOp : PylirPy_Op<"mroLookup", [
  MemoryEffects<[MemRead<ObjectResource>]>, NoCaptures]> {
  let arguments = (ins DynamicType:$mro_tuple, Index:$slot);
  let results = (outs DynamicType:$result);

  let assemblyFormat = "$slot `in` $mro_tuple attr-dict";

  let description = [{
    This op does a linear search through the given tuple, accessing the slot on
    every element and returning the first one bound, not accessing any objects
    after.
    If `$mro_tuple` is not really a tuple (or subclass of) the behaviour is
    undefined.
    If `$slot` is greater or equal to the amount of slots in an element accessed
    during traversal, the behaviour is undefined.
    If a bound slot was found it is returned to `$result`, else it is unbound.
  }];

  let hasFolder = 1;
}

def PylirPy_UnpackOp : PylirPy_Op<"unpack", [AlwaysBound,
  AddableExceptionHandling<"UnpackExOp">, AttrSizedResultSegments]> {

  let arguments = (ins DynamicType:$iterable);

  let results = (outs Variadic<DynamicType>:$before,
    Optional<DynamicType>:$rest, Variadic<DynamicType>:$after);

  let builders = [
    OpBuilder<(ins "std::size_t":$count,
                   "std::optional<std::size_t>":$restIndex,
                   "mlir::Value":$iterable)>
  ];

  let assemblyFormat = [{
      $iterable `:` `(` ( `)` ) : ( `` type($before)^ `)`)? (`,` type($rest)^
        (`,` `(` type($after)^ `)`)?)? attr-dict
  }];

  let description = [{
    Takes an iterable object and unpacks it into multiple values.
    The results of this operation may consist of one or three result parts.

    One part form example:
    ```
    %before:3 = unpack %iterable : (!py.dynamic, !py.dynamic, !py.dynamic)
    ```
    This expects the iterable to contain 3 elements to unpack them into
    %before:0 to 2 respectively.

    Three part form example:
    ```
    %before:3, %rest, %after:2 = unpack %iterable : (!py.dynamic, !py.dynamic, !py.dynamic), !py.dynamic, (!py.dynamic, !py.dynamic)
    ```

    This expects the iterable to contain at least 5 elements.
    The first 3  elements are unpacked into %before:0 to 2, and the last 2
    elements of the  iterable into %after:0 to 1.
    All other left over elements will form a list that is assigned to %rest.

    The iterable's iterator is created via a call to pythons builtin `iter`
    function, and the elements extracted via pythons builtin `next` function.
    Exceptions emitted by either of these methods, with the exception of
    `StopIteration`, are propagated as is.
    If the iterable contains too few or too many elements, a `ValueError` is
    raised.
  }];

  let hasVerifier = 1;
}

def PylirPy_UnpackExOp : CreateExceptionHandlingVariant<PylirPy_UnpackOp> {
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "std::size_t":$count,
            "std::optional<std::size_t>":$restIndex,
            "mlir::Value":$iterable,
            "mlir::Block*":$happy_path,
            "mlir::ValueRange":$normal_dest_operands,
            "mlir::Block*":$unwindPath,
            "mlir::ValueRange":$unwind_dest_operands)>
  ];
}

//===----------------------------------------------------------------------===//
// Global Ops
//===----------------------------------------------------------------------===//

def PylirPy_ExternalOp : PylirPy_Op<"external", [Symbol]> {
  let arguments = (ins SymbolNameAttr:$sym_name, PylirPy_GlobalValueAttr:$attr);

  let description = [{
    This op is used to mark a `#py.globalValue` as external.
    This may have one of two semantics, depending on whether the
    `#py.globalValue` has an initializer:
    * If it has an initializer, the attribute is visible from other modules
      importing it using a `py.external` op with the same name and a
      `#py.globalValue` without an initializer.
      This can also be used from other programming languages to import the
      global value given a matching target dependent memory layout.
    * Otherwise, the `#py.globalValue` is referencing a `#py.globalValue` with
      an initializer marked external in another module, or a `#py.globalValue`
      defined in another programming language.
      There must be exactly one such `#py.globalValue` in the program.

    A `#py.globalValue` may only occur in a `py.external` once.
  }];

  let assemblyFormat = [{
    $sym_name `,` $attr attr-dict
  }];
}

defvar GlobalOpSupportedTypes = AnyTypeOf<[DynamicType, Index, F64]>;
defvar GlobalOpSupportedAttrs = AnyAttrOf<[ObjectAttrInterface, IndexAttr,
  F64Attr, PylirPy_GlobalValueAttr]>;

def PylirPy_GlobalOp : PylirPy_Op<"global", [Symbol, VerifyAttributesOpTrait]> {

  let arguments = (ins
      SymbolNameAttr:$sym_name,
      OptionalAttr<StrAttr>:$sym_visibility,
      TypeAttrOf<GlobalOpSupportedTypes>:$type,
      OptionalAttr<GlobalOpSupportedAttrs>:$initializer);

  let results = (outs);

  let assemblyFormat = [{
    ($sym_visibility^)? $sym_name `:` $type (`=` $initializer^)? attr-dict
  }];

  let description = [{
    This op creates a global to which values of type `$type` can be assigned.
    See `py.store` and `py.load` how to use it.
    The global may optionally have an initializer which serves as the initial
    value returned by `py.load` if no value has ever been stored into the
    global.
    If the global does not have an initializer loading from it before a value
    has ever been stored in it is undefined behaviour.
  }];

  let hasVerifier = 1;
}

def PylirPy_LoadOp : PylirPy_Op<"load", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<MemoryFoldInterface>]> {
  let summary = "load operation";

  let arguments = (ins
    Arg<FlatSymbolRefAttr, "", [MemRead<GlobalResource>]>:$global);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$global `:` type($result) attr-dict";

  let builders = [
    OpBuilder<(ins "::pylir::Py::GlobalOp":$global), [{
      build($_builder, $_state, global.getType(),
        ::mlir::FlatSymbolRefAttr::get(global));
    }]>
  ];

  let description = [{
    This op retrieves the value that was last stored into `$global`.
    If no value was previously stored into `$global` the behaviour is undefined.
  }];
}

def PylirPy_StoreOp : PylirPy_Op<"store", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {

  let summary = "store operation";

  let arguments = (ins AnyType:$value,
    Arg<FlatSymbolRefAttr, "", [MemWrite<GlobalResource>]>:$global);
  let results = (outs);

  let assemblyFormat = [{
    $value `:` type($value) `into` $global attr-dict
  }];

  let description = [{
    This op binds the value denoted by `$value` to the `$global`.
    Subsequent loads will return the last stored values.
    The type of `$value` has to match up exactly with the type of `$global`.
  }];
}

//===----------------------------------------------------------------------===//
// (Inter-procedural) Control flow Ops
//===----------------------------------------------------------------------===//

def PylirPy_CallOp : PylirPy_Op<"call", [AlwaysBound,
  AddableExceptionHandling<"InvokeOp">,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "call operation";

  let arguments = (ins FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$call_operands);
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    $callee `(` $call_operands `)` attr-dict `:` functional-type($call_operands, results)
  }];

  let builders = [
    OpBuilder<(ins "::mlir::FunctionOpInterface":$callee,
                   CArg<"::mlir::ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", ::mlir::SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getResultTypes());
    }]>,
    OpBuilder<(ins "::mlir::TypeRange":$results, "::mlir::StringAttr":$callee,
      CArg<"::mlir::ValueRange", "{}">:$operands), [{
      build($_builder, $_state, results, ::mlir::SymbolRefAttr::get(callee),
            operands);
    }]>,
    OpBuilder<(ins  "::mlir::TypeRange":$results, "::llvm::StringRef":$callee,
      CArg<"::mlir::ValueRange", "{}">:$operands), [{
      build($_builder, $_state, results,
            ::mlir::StringAttr::get($_builder.getContext(), callee), operands);
    }]>];
}

def PylirPy_InvokeOp
  : CreateExceptionHandlingVariant<PylirPy_CallOp, "invoke"> {
  let summary = "invoke operation";
}

def PylirPy_FuncOp : PylirPy_Op<"func", [FunctionOpInterface, IsolatedFromAbove,
  OpAsmOpInterface, AutomaticAllocationScope, CallableOpInterface,
  EntryArgsBound, VerifyAttributesOpTrait]> {
  let arguments = (ins StrAttr:$sym_name,
           TypeAttrOf<FunctionType>:$function_type,
           OptionalAttr<StrAttr>:$sym_visibility,
           OptionalAttr<DictArrayAttr>:$arg_attrs,
           OptionalAttr<DictArrayAttr>:$res_attrs);

  let regions = (region AnyRegion:$body);

  let builders = [
    OpBuilder<(ins "llvm::Twine":$name, "mlir::FunctionType":$type), [{
      build($_builder, $_state, $_builder.getStringAttr(name),
            mlir::TypeAttr::get(type), nullptr, nullptr, nullptr);
    }]>
  ];

  let extraClassDeclaration = [{

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Implementation
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    llvm::ArrayRef<mlir::Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }

    /// Returns the result types of this function.
    llvm::ArrayRef<mlir::Type> getResultTypes() {
      return getFunctionType().getResults();
    }

    static FuncOp
    create(mlir::Location loc, llvm::Twine name, mlir::FunctionType type) {
      mlir::OpBuilder builder(loc.getContext());
      return builder.create<FuncOp>(loc, name, type);
    }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Implementation
    //===------------------------------------------------------------------===//

    static llvm::StringRef getDefaultDialect() {
      return pylir::Py::PylirPyDialect::getDialectNamespace();
    }

    //===------------------------------------------------------------------===//
    // CallableOpInterface Implementation
    //===------------------------------------------------------------------===//

    mlir::Region *getCallableRegion() {
      return isExternal() ? nullptr : &getBody();
    }

    llvm::ArrayRef<mlir::Type> getCallableResults() {
      return getFunctionType().getResults();
    }

    mlir::ArrayAttr getCallableArgAttrs() {
      return getArgAttrs().value_or(nullptr);
    }

    mlir::ArrayAttr getCallableResAttrs() {
      return getResAttrs().value_or(nullptr);
    }

    //===------------------------------------------------------------------===//
    // Symbol Implementation
    //===------------------------------------------------------------------===//

    bool isDeclaration() {
      return isExternal();
    }
  }];

  let hasCustomAssemblyFormat = 1;
}

def PylirPy_ReturnOp : PylirPy_Op<"return", [HasParent<"FuncOp">, Terminator,
  ReturnLike]> {
  let arguments = (ins Variadic<AnyType>:$arguments);

  let assemblyFormat = "($arguments^ `:` type($arguments))? attr-dict";

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, mlir::ValueRange{});
    }]>
  ];

  let hasVerifier = 1;
}

def PylirPy_UnreachableOp : PylirPy_Op<"unreachable", [Terminator]> {
  let results = (outs);

  let assemblyFormat = "attr-dict";
}

def PylirPy_RaiseOp : PylirPy_Op<"raise", [Terminator,
  AddableExceptionHandling<"RaiseExOp">]> {
  let summary = "raise exception";

  let arguments = (ins DynamicType:$exception);
  let results = (outs);

  let assemblyFormat = [{
    $exception attr-dict
  }];
}

def PylirPy_RaiseExOp : CreateExceptionHandlingVariant<PylirPy_RaiseOp>;

#endif
