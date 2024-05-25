// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_PY_TO_PYLIR_MEM_TABLEGEN
#define PYLIR_PY_TO_PYLIR_MEM_TABLEGEN

include "pylir/Optimizer/PylirPy/IR/PylirPyOps.td"
include "pylir/Optimizer/PylirMem/IR/PylirMemOps.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/IR/PatternBase.td"

defvar TupleRef = ConstantStrAttr<PylirPy_GlobalValueAttr, "builtins.tuple">;
defvar TypeRef = ConstantStrAttr<PylirPy_GlobalValueAttr, "builtins.type">;
defvar WrapOverflowFlag = ConstantAttr<Arith_IntegerOverflowAttr,
  "::mlir::arith::IntegerOverflowFlags::none">;

def : Pattern<(PylirPy_MakeTupleOp $args, $_), [
  (PylirPy_ConstantOp:$tupleType TupleRef),
  (Arith_ConstantOp:$size (NativeCodeCall<"$_builder.getIndexAttr($0.size())">
    $args)),
  (PylirMem_GCAllocObjectOp:$memory $tupleType, $size),
  (PylirMem_InitTupleOp $memory, $args)
]>;

def : Pattern<(PylirPy_TuplePrependOp $input, $tuple), [
  (PylirPy_ConstantOp:$tupleType TupleRef),
  (PylirPy_TupleLenOp:$prevSize $tuple),
  (Arith_AddIOp:$newSize $prevSize, (Arith_ConstantOp ConstantAttr<IndexAttr,
    "1">), WrapOverflowFlag),
  (PylirMem_GCAllocObjectOp:$memory $tupleType, $newSize),
  (PylirMem_InitTuplePrependOp $memory, $input, $tuple)
]>;

def : Pattern<(PylirPy_TupleDropFrontOp $count, $tuple), [
  (PylirPy_ConstantOp:$tupleType TupleRef),
  (PylirPy_TupleLenOp:$prevSize $tuple),
  (Arith_SubIOp:$newSize $prevSize, $count, WrapOverflowFlag),
  (PylirMem_GCAllocObjectOp:$memory $tupleType, $newSize),
  (PylirMem_InitTupleDropFrontOp $memory, $count, $tuple)
]>;

def : Pattern<(PylirPy_TupleCopyOp $tuple, $typeObj), [
  (PylirPy_TupleLenOp:$size $tuple),
  (PylirMem_GCAllocObjectOp:$memory $typeObj, $size),
  (PylirMem_InitTupleCopyOp $memory, $tuple)
]>;

def : Pat<(PylirPy_ListToTupleOp $list), (PylirMem_InitTupleFromListOp
  (PylirMem_GCAllocObjectOp (PylirPy_ConstantOp TupleRef),
    (PylirPy_ListLenOp $list)), $list)>;

class GCWithSlots<ConstantAttr type_object> {
  dag result = (PylirMem_GCAllocObjectOp (PylirPy_ConstantOp:$c type_object),
                       (PylirPy_TupleLenOp (PylirPy_TypeSlotsOp $c)));
}

defvar ListRef = ConstantStrAttr<PylirPy_GlobalValueAttr, "builtins.list">;

def : Pat<(PylirPy_MakeListOp $args, $_),
  (PylirMem_InitListOp GCWithSlots<ListRef>.result, $args)>;

defvar SetRef = ConstantStrAttr<PylirPy_GlobalValueAttr, "builtins.set">;

def : Pat<(PylirPy_MakeSetOp $args, $_),
  (PylirMem_InitSetOp GCWithSlots<SetRef>.result, $args)>;

defvar StrRef = ConstantStrAttr<PylirPy_GlobalValueAttr, "builtins.str">;

def : Pat<(PylirPy_StrConcatOp $args),
  (PylirMem_InitStrOp GCWithSlots<StrRef>.result, $args)>;

def : Pat<(PylirPy_StrCopyOp $arg, $typeObj),
  (PylirMem_InitStrOp (PylirMem_GCAllocObjectOp $typeObj, (PylirPy_TupleLenOp
    (PylirPy_TypeSlotsOp $typeObj))),
    (NativeCodeCall<"::llvm::SmallVector<::mlir::Value>{$0}"> $arg))>;

def : Pat<(PylirPy_IntToStrOp $integer),
  (PylirMem_InitStrFromIntOp GCWithSlots<StrRef>.result, $integer)>;

defvar FunctionRef = ConstantStrAttr<PylirPy_GlobalValueAttr, "builtins.function">;

def : Pat<(PylirPy_MakeFuncOp $symbol, $closure_args),
  (PylirMem_InitFuncOp (PylirMem_GCAllocFunctionOp (NativeCodeCall<[{
    $_builder.getTypeArrayAttr($0.getTypes())
  }]> $closure_args)), $symbol,
    $closure_args)>;

defvar BoolRef = ConstantStrAttr<PylirPy_GlobalValueAttr, "builtins.bool">;

defvar IndexSizedInteger = NativeCodeCall<[{
  $_builder.getIntegerType(mlir::DataLayout::closest($0.getOwner())
    .getTypeSizeInBits($_builder.getIndexType()))
}]>;

def : Pattern<(PylirPy_BoolFromI1Op:$op $input), [
  (Arith_ExtUIOp:$boolToInt $input, (returnType (IndexSizedInteger $op))),
  (Arith_IndexCastOp:$intToIndex $boolToInt,
    (returnType "$_builder.getIndexType()")),
  (PylirMem_InitIntUnsignedOp GCWithSlots<BoolRef>.result, $intToIndex)
]>;

defvar IntRef = ConstantStrAttr<PylirPy_GlobalValueAttr, "builtins.int">;

def : Pat<(PylirPy_IntFromSignedOp $input), (PylirMem_InitIntSignedOp
  GCWithSlots<IntRef>.result, $input)>;

def : Pat<(PylirPy_IntFromUnsignedOp $input), (PylirMem_InitIntUnsignedOp
  GCWithSlots<IntRef>.result, $input)>;

def : Pat<(PylirPy_IntAddOp $lhs, $rhs), (PylirMem_InitIntAddOp
  GCWithSlots<IntRef>.result, $lhs, $rhs)>;

defvar FloatRef = ConstantStrAttr<PylirPy_GlobalValueAttr, "builtins.float">;

def : Pat<(PylirPy_FloatFromF64 $input),
  (PylirMem_InitFloatOp GCWithSlots<FloatRef>.result, $input)>;


def : Pat<(PylirPy_MakeObjectOp $typeObj),
  (PylirMem_InitObjectOp (PylirMem_GCAllocObjectOp $typeObj,
    (PylirPy_TupleLenOp (PylirPy_TypeSlotsOp $typeObj))))>;

def : Pat<(PylirPy_MakeTypeOp $name, $mro_tuple, $slots_tuple),
  (PylirMem_InitTypeOp GCWithSlots<TypeRef>.result, $name,
    (PylirMem_GCAllocObjectOp (PylirPy_ConstantOp TupleRef),
      (Arith_AddIOp (PylirPy_TupleLenOp $mro_tuple),
        (Arith_ConstantOp ConstantAttr<IndexAttr, "1">),
          WrapOverflowFlag)),
    $mro_tuple, $slots_tuple)>;

#endif
