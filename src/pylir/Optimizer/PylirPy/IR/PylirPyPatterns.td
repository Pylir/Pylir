
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

def : Pat<(PylirPy_TuplePrependOp $input, (ConstantLikeMatcher PylirPy_TupleAttr:$args)),
          (prependTupleConst $input, $args)>;

def : Pattern<(PylirPy_GetSlotOp:$loc $object, (PylirPy_TypeOfOp (PylirPy_TypeOfOp $_)), IsTypeSlot:$slot),
    [(PylirPy_ConstantOp:$constant TypeRef), (PylirPy_GetSlotOp $object, $constant, $slot, (location $loc))]
>;

def : Pattern<(PylirPy_SetSlotOp:$loc $object, (PylirPy_TypeOfOp (PylirPy_TypeOfOp $_)), IsTypeSlot:$slot, $value),
    [(PylirPy_ConstantOp:$constant TypeRef), (PylirPy_SetSlotOp $object, $constant, $slot, $value, (location $loc))]
>;

defvar Arith_CmpIEq = !head(!filter(x, Arith_CmpIPredicateAttr.enumerants, !eq(x.symbol, "eq")));

def : Pat<(Arith_CmpIOp Arith_CmpIEq, (PylirPy_ObjectIdOp $lhs), (PylirPy_ObjectIdOp $rhs)), (PylirPy_IsOp $lhs, $rhs)>;

class IntegerWidthComp<string op> : Constraint<CPred<
    "$0.getType().getIntOrFloatBitWidth() " # op # "$1.getType().getIntOrFloatBitWidth()">,
     "integer width is " # op # " than the other">;

def TypesMatch : Constraint<CPred<"$0.getType() == $1.getType()">, "has same type">;

def : Pattern<(PylirPy_IntToIntegerOp:$result (PylirPy_IntFromIntegerOp AnyInteger:$input)),
                [(Arith_TruncIOp $input), (Arith_ConstantOp ConstBoolAttrTrue)],
                [(AnyInteger $result), (IntegerWidthComp<">"> $input, $result__0)]>;

def : Pattern<(PylirPy_IntToIntegerOp:$result (PylirPy_IntFromIntegerOp AnyInteger:$input)),
                [(Arith_ExtSIOp $input), (Arith_ConstantOp ConstBoolAttrTrue)],
                [(AnyInteger $result), (IntegerWidthComp<"<"> $input, $result__0)]>;

def : Pattern<(PylirPy_IntToIntegerOp:$result (PylirPy_IntFromIntegerOp $input)),
                [(replaceWithValue $input), (Arith_ConstantOp ConstBoolAttrTrue)],
                [(TypesMatch $result__0, $input)]>;


#endif
