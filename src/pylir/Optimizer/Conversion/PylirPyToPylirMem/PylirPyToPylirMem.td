// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_PY_TO_PYLIR_MEM_TABLEGEN
#define PYLIR_PY_TO_PYLIR_MEM_TABLEGEN

include "pylir/Optimizer/PylirPy/IR/PylirPyOps.td"
include "pylir/Optimizer/PylirMem/IR/PylirMemOps.td"
include "mlir/Dialect/Arithmetic/IR/ArithmeticOps.td"
include "mlir/IR/PatternBase.td"

defvar TupleRef = ConstantStrAttr<FlatSymbolRefAttr, "builtins.tuple">;
defvar TypeRef = ConstantStrAttr<FlatSymbolRefAttr, "builtins.type">;

def : Pattern<(PylirPy_MakeTupleOp $args, $_), [
    (PylirPy_ConstantOp:$tupleType TupleRef),
    (Arith_ConstantOp:$size (NativeCodeCall<"$_builder.getIndexAttr($0.size())"> $args)),
    (PylirMem_GCAllocTupleOp:$memory $tupleType, $size),
    (PylirMem_InitTupleOp $memory, $args)
]>;

def : Pattern<(PylirPy_TuplePrependOp $input, $tuple), [
    (PylirPy_ConstantOp:$tupleType TupleRef),
    (PylirPy_TupleLenOp:$prevSize $tuple),
    (Arith_AddIOp:$newSize $prevSize, (Arith_ConstantOp ConstantAttr<IndexAttr, "1">)),
    (PylirMem_GCAllocTupleOp:$memory $tupleType, $newSize),
    (PylirMem_InitTuplePrependOp $memory, $input, $tuple)
]>;

def : Pattern<(PylirPy_TupleDropFrontOp $count, $tuple), [
    (PylirPy_ConstantOp:$tupleType TupleRef),
    (PylirPy_TupleLenOp:$prevSize $tuple),
    (Arith_SubIOp:$newSize $prevSize, $count),
    (PylirMem_GCAllocTupleOp:$memory $tupleType, $newSize),
    (PylirMem_InitTupleDropFrontOp $memory, $count, $tuple)
]>;

def : Pattern<(PylirPy_TupleCopyOp $tuple, $typeObj), [
    (PylirPy_TupleLenOp:$size $tuple),
    (PylirMem_GCAllocTupleOp:$memory $typeObj, $size),
    (PylirMem_InitTupleCopyOp $memory, $tuple)
]>;

def : Pat<(PylirPy_ListToTupleOp $list), (PylirMem_InitTupleFromListOp
    (PylirMem_GCAllocTupleOp (PylirPy_ConstantOp TupleRef), (PylirPy_ListLenOp $list)), $list)>;

defvar ListRef = ConstantStrAttr<FlatSymbolRefAttr, "builtins.list">;

def : Pat<(PylirPy_MakeListOp $args, $_),
    (PylirMem_InitListOp (PylirMem_GCAllocObjectOp (PylirPy_ConstantOp ListRef)), $args)>;

defvar SetRef = ConstantStrAttr<FlatSymbolRefAttr, "builtins.set">;

def : Pat<(PylirPy_MakeSetOp $args, $_),
    (PylirMem_InitSetOp (PylirMem_GCAllocObjectOp (PylirPy_ConstantOp SetRef)), $args)>;

defvar StrRef = ConstantStrAttr<FlatSymbolRefAttr, "builtins.str">;

def : Pat<(PylirPy_StrConcatOp $args),
    (PylirMem_InitStrOp (PylirMem_GCAllocObjectOp (PylirPy_ConstantOp StrRef)), $args)>;

def : Pat<(PylirPy_StrCopyOp $arg, $typeObj),
    (PylirMem_InitStrOp (PylirMem_GCAllocObjectOp $typeObj),
        (NativeCodeCall<"::llvm::SmallVector<::mlir::Value>{$0}"> $arg))>;

def : Pat<(PylirPy_IntToStrOp $integer),
    (PylirMem_InitStrFromIntOp (PylirMem_GCAllocObjectOp (PylirPy_ConstantOp StrRef)), $integer)>;

defvar FunctionRef = ConstantStrAttr<FlatSymbolRefAttr, "builtins.function">;

def : Pat<(PylirPy_MakeFuncOp $symbol),
    (PylirMem_InitFuncOp (PylirMem_GCAllocObjectOp (PylirPy_ConstantOp FunctionRef)), $symbol)>;

defvar BoolRef = ConstantStrAttr<FlatSymbolRefAttr, "builtins.bool">;

defvar IndexSizedInteger = NativeCodeCall<[{
    $_builder.getIntegerType(mlir::DataLayout::closest($0.getOwner()).getTypeSizeInBits($_builder.getIndexType()))
}]>;

def : Pattern<(PylirPy_BoolFromI1Op:$op $input), [
    (PylirMem_GCAllocObjectOp:$boolMem (PylirPy_ConstantOp BoolRef)),
    (Arith_ExtUIOp:$boolToInt $input, (returnType (IndexSizedInteger $op))),
    (Arith_IndexCastOp:$intToIndex $boolToInt, (returnType "$_builder.getIndexType()")),
    (PylirMem_InitIntUnsignedOp $boolMem, $intToIndex)
]>;

defvar IntRef = ConstantStrAttr<FlatSymbolRefAttr, "builtins.int">;

def : Pat<(PylirPy_IntFromSignedOp $input), (PylirMem_InitIntSignedOp
    (PylirMem_GCAllocObjectOp (PylirPy_ConstantOp IntRef)), $input)>;

def : Pat<(PylirPy_IntFromUnsignedOp $input), (PylirMem_InitIntUnsignedOp
    (PylirMem_GCAllocObjectOp (PylirPy_ConstantOp IntRef)), $input)>;

def : Pat<(PylirPy_IntAddOp $lhs, $rhs), (PylirMem_InitIntAddOp
    (PylirMem_GCAllocObjectOp (PylirPy_ConstantOp IntRef)), $lhs, $rhs)>;

def : Pat<(PylirPy_MakeObjectOp $typeObj), (PylirMem_InitObjectOp (PylirMem_GCAllocObjectOp $typeObj))>;

#endif
