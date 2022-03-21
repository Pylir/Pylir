
#ifndef PYLIR_PY_TO_PYLIR_MEM_TABLEGEN
#define PYLIR_PY_TO_PYLIR_MEM_TABLEGEN

include "pylir/Optimizer/PylirPy/IR/PylirPyOps.td"
include "pylir/Optimizer/PylirMem/IR/PylirMemOps.td"
include "mlir/Dialect/Arithmetic/IR/ArithmeticOps.td"
include "mlir/IR/PatternBase.td"

defvar TupleRef = ConstantStrAttr<FlatSymbolRefAttr, "builtins.tuple">;
defvar TypeRef = ConstantStrAttr<FlatSymbolRefAttr, "builtins.type">;
defvar TypeClass = "$_builder.getType<::pylir::Py::ClassType>(" # !subst("$0", TypeRef.value, TypeRef.attr.constBuilderCall) # ", llvm::None)";

def : Pattern<(PylirPy_MakeTupleOp $args, $_), [
    (PylirPy_ConstantOp:$tupleType TupleRef, (returnType TypeClass)),
    (Arith_ConstantOp:$size (NativeCodeCall<"$_builder.getIndexAttr($0.size())"> $args)),
    (PylirMem_GCAllocTupleOp:$memory $tupleType, $size),
    (PylirMem_InitTupleOp $memory, $args)
]>;

def : Pattern<(PylirPy_TuplePrependOp $input, $tuple), [
    (PylirPy_ConstantOp:$tupleType TupleRef, (returnType TypeClass)),
    (PylirPy_TupleLenOp:$prevSize $tuple),
    (Arith_AddIOp:$newSize $prevSize, (Arith_ConstantOp ConstantAttr<IndexAttr, "1">)),
    (PylirMem_GCAllocTupleOp:$memory $tupleType, $newSize),
    (PylirMem_InitTuplePrependOp $memory, $input, $tuple)
]>;

def : Pattern<(PylirPy_TuplePopFrontOp $tuple), [
    (PylirPy_ConstantOp:$tupleType TupleRef, (returnType TypeClass)),
    (PylirPy_TupleLenOp:$prevSize $tuple),
    (Arith_SubIOp:$newSize $prevSize, (Arith_ConstantOp ConstantAttr<IndexAttr, "1">)),
    (PylirMem_GCAllocTupleOp:$memory $tupleType, $newSize),
    (PylirPy_TupleGetItemOp:$element $tuple, (Arith_ConstantOp ConstantAttr<IndexAttr, "0">)),
    (PylirMem_InitTuplePopFrontOp $memory, $tuple)
]>;

def : Pat<(PylirPy_ListToTupleOp $list), (PylirMem_InitTupleFromListOp
    (PylirMem_GCAllocTupleOp (PylirPy_ConstantOp TupleRef, (returnType TypeClass)), (PylirPy_ListLenOp $list)), $list)>;

defvar ListRef = ConstantStrAttr<FlatSymbolRefAttr, "builtins.list">;

def : Pat<(PylirPy_MakeListOp $args, $_),
    (PylirMem_InitListOp (PylirMem_GCAllocObjectOp (PylirPy_ConstantOp ListRef, (returnType TypeClass))), $args)>;

defvar SetRef = ConstantStrAttr<FlatSymbolRefAttr, "builtins.set">;

def : Pat<(PylirPy_MakeSetOp $args, $_),
    (PylirMem_InitSetOp (PylirMem_GCAllocObjectOp (PylirPy_ConstantOp SetRef, (returnType TypeClass))), $args)>;

defvar StrRef = ConstantStrAttr<FlatSymbolRefAttr, "builtins.str">;

def : Pat<(PylirPy_StrConcatOp $args),
    (PylirMem_InitStrOp (PylirMem_GCAllocObjectOp (PylirPy_ConstantOp StrRef, (returnType TypeClass))), $args)>;

def : Pat<(PylirPy_StrCopyOp $arg, $typeObj),
    (PylirMem_InitStrOp (PylirMem_GCAllocObjectOp $typeObj),
        (NativeCodeCall<"::llvm::SmallVector<::mlir::Value>{$0}"> $arg))>;

def : Pat<(PylirPy_IntToStrOp $integer),
    (PylirMem_InitStrFromIntOp (PylirMem_GCAllocObjectOp (PylirPy_ConstantOp StrRef, (returnType TypeClass))), $integer)>;

defvar FunctionRef = ConstantStrAttr<FlatSymbolRefAttr, "builtins.function">;

def : Pat<(PylirPy_MakeFuncOp $symbol),
    (PylirMem_InitFuncOp (PylirMem_GCAllocObjectOp (PylirPy_ConstantOp FunctionRef, (returnType TypeClass))), $symbol)>;

defvar BoolRef = ConstantStrAttr<FlatSymbolRefAttr, "builtins.bool">;

def : Pat<(PylirPy_BoolFromI1Op $input), (PylirMem_InitIntOp
    (PylirMem_GCAllocObjectOp (PylirPy_ConstantOp BoolRef, (returnType TypeClass))), $input)>;

defvar IntRef = ConstantStrAttr<FlatSymbolRefAttr, "builtins.int">;

def : Pat<(PylirPy_IntFromIntegerOp $input), (PylirMem_InitIntOp
    (PylirMem_GCAllocObjectOp (PylirPy_ConstantOp IntRef, (returnType TypeClass))), $input)>;

def : Pat<(PylirPy_IntAddOp $lhs, $rhs), (PylirMem_InitIntAddOp
    (PylirMem_GCAllocObjectOp (PylirPy_ConstantOp IntRef, (returnType TypeClass))), $lhs, $rhs)>;

def : Pat<(PylirPy_MakeObjectOp $typeObj), (PylirMem_InitObjectOp (PylirMem_GCAllocObjectOp $typeObj))>;

#endif
