
#ifndef PYLIR_PY_PATTERNS_TABLEGEN
#define PYLIR_PY_PATTERNS_TABLEGEN

include "mlir/IR/OpBase.td"
include "mlir/Dialect/Arithmetic/IR/ArithmeticOps.td"
include "pylir/Optimizer/PylirPy/IR/PylirPyOps.td"

def prependTuple : NativeCodeCall<"prependTuple($_builder, $_loc, $0, $1, $2)">;

def prependTupleConst : NativeCodeCall<"prependTupleConst($_builder, $_loc, $0, $1)">;

def IsTypeSlot : AttrConstraint<
    CPred<"$_self.isa<::mlir::StringAttr>() && ::isTypeSlot($_self.cast<::mlir::StringAttr>().getValue())">,
    "is type slot">;

defvar TypeRef = ConstantStrAttr<FlatSymbolRefAttr, "builtins.type">;

def : Pat<(PylirPy_TuplePrependOp $input, (PylirPy_MakeTupleOp $args, $expansion)),
          (prependTuple $input, $args, $expansion)>;

def : Pat<(PylirPy_TuplePrependOp $input, (PylirPy_ConstantOp PylirPy_TupleAttr:$args)),
          (prependTupleConst $input, $args)>;

def : Pattern<(PylirPy_GetSlotOp $object, (PylirPy_TypeOfOp (PylirPy_TypeOfOp $_)), IsTypeSlot:$slot),
    [(PylirPy_ConstantOp:$constant TypeRef), (PylirPy_GetSlotOp $object, $constant, $slot)]
>;

def : Pattern<(PylirPy_SetSlotOp $object, (PylirPy_TypeOfOp (PylirPy_TypeOfOp $_)), IsTypeSlot:$slot, $value),
    [(PylirPy_ConstantOp:$constant TypeRef), (PylirPy_SetSlotOp $object, $constant, $slot, $value)]
>;

defvar Arith_CmpIEq = !head(!filter(x, Arith_CmpIPredicateAttr.enumerants, !eq(x.symbol, "eq")));

def : Pat<(Arith_CmpIOp Arith_CmpIEq, (PylirPy_ObjectIdOp $lhs), (PylirPy_ObjectIdOp $rhs)), (PylirPy_IsOp $lhs, $rhs)>;

#endif
