// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_PY_PATTERNS_TABLEGEN
#define PYLIR_PY_PATTERNS_TABLEGEN

include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "pylir/Optimizer/PylirPy/IR/PylirPyOps.td"

class ResolvesTo<int constOnly = 1>
	: NativeCodeCall<"::resolvesToPattern($_self, $0, " # !if(constOnly, "true", "false") # ")">;

def prependTupleConst : NativeCodeCall<"prependTupleConst($_builder, $_loc, $0, $1)">;

def IsTypeSlot : AttrConstraint<CPred<"::isTypeSlot($_self.cast<::mlir::StringAttr>().getValue())">, "is type slot">;

class DenseArrayCount<int n> : AttrConstraint<
    CPred<"$_self.cast<::mlir::DenseArrayAttr>().size() == " #n>,
    "with exactly " # n # " elements">;

defvar TypeRef = ConstantStrAttr<PylirPy_RefAttr, "builtins.type">;

def : Pat<(PylirPy_TuplePrependOp $input, (PylirPy_MakeTupleOp $args, DenseArrayCount<0>)),
          (PylirPy_MakeTupleOp
            (NativeCodeCall<[{ [&]{ auto vector = llvm::to_vector($1);
                                    vector.insert(vector.begin(), $0);
                                    return vector; }() }]> $input, $args), ConstantAttr<DenseI32ArrayAttr, "{}">)>;

def : Pat<(PylirPy_TuplePrependOp $input, (ResolvesTo<0> PylirPy_TupleAttr:$args)),
          (prependTupleConst $input, $args)>;

def : Pat<(PylirPy_TupleDropFrontOp (ConstantLikeMatcher IndexAttr:$count),
									(PylirPy_MakeTupleOp $args, DenseArrayCount<0>)),
          (PylirPy_MakeTupleOp
            (NativeCodeCall<[{ [&]{
                                    auto value = $0.cast<mlir::IntegerAttr>().getValue().getZExtValue();
                                    auto vector = llvm::to_vector($1);
                                    if (vector.size() <= value)
                                    {
                                        vector.clear();
                                    }
                                    else
                                    {
                                        vector.erase(vector.begin(), vector.begin() + value);
                                    }
                                    return vector;
                                }() }]> $count, $args), ConstantAttr<DenseI32ArrayAttr, "{}">)>;

class BinOpAttrs<string op, Attr attrType, Attr resultAttr = attrType>
	: NativeCodeCall<!subst("$0",
		!subst("$_self", "$0.cast<" # attrType.storageType # ">()", attrType.convertFromStorage)
		# " " # op # " " # !subst("$_self", "$1.cast<" # attrType.storageType # ">()", attrType.convertFromStorage),
		resultAttr.constBuilderCall)>;

class PlusAttrs<Attr attrType, Attr resultAttr = attrType> : BinOpAttrs<"+", attrType, resultAttr>;

def : Pat<(PylirPy_TupleDropFrontOp $count1, (PylirPy_TupleDropFrontOp $count2, $tuple)),
          (PylirPy_TupleDropFrontOp (Arith_AddIOp $count1, $count2), $tuple)>;

defvar TupleRef = ConstantStrAttr<PylirPy_RefAttr, "builtins.tuple">;

def : Pat<(PylirPy_TupleDropFrontOp (ConstantLikeMatcher ConstantAttr<IndexAttr,"0">), $tuple),
			(PylirPy_TupleCopyOp $tuple, (PylirPy_ConstantOp TupleRef))>;

def : Pat<(PylirPy_TupleDropFrontOp (ConstantLikeMatcher ConstantAttr<IndexAttr,"1">),
				(PylirPy_TuplePrependOp $_, $tuple)), (PylirPy_TupleCopyOp $tuple, (PylirPy_ConstantOp TupleRef))>;

defvar Arith_CmpIEq = !head(!filter(x, Arith_CmpIPredicateAttr.enumerants, !eq(x.symbol, "eq")));
defvar Arith_CmpINe = !head(!filter(x, Arith_CmpIPredicateAttr.enumerants, !eq(x.symbol, "ne")));

def : Pat<(Arith_CmpIOp Arith_CmpIEq, (PylirPy_ObjectIdOp $lhs), (PylirPy_ObjectIdOp $rhs)), (PylirPy_IsOp $lhs, $rhs)>;
def : Pat<(Arith_CmpIOp Arith_CmpINe, (PylirPy_ObjectIdOp $lhs), (PylirPy_ObjectIdOp $rhs)),
			(Arith_XOrIOp (PylirPy_IsOp $lhs, $rhs), (Arith_ConstantOp ConstBoolAttrTrue))>;

def IsConstant : Constraint<CPred<"::mlir::matchPattern($0, ::mlir::m_Constant())">, "is constant">;

class NotConstraint<Constraint con> : Constraint<Neg<con.predicate>, "not " # con.summary>;

def : Pat<(PylirPy_IntCmpOp $kind, $lhs, $rhs),
        (PylirPy_IntCmpOp (NativeCodeCall<"reversePredicate($0)"> $kind), $rhs, $lhs),
        [(IsConstant $lhs), (NotConstraint<IsConstant> $rhs)]>;

def : Pat<(Arith_XOrIOp (PylirPy_IntCmpOp $pred, $lhs, $rhs), (ConstantLikeMatcher ConstBoolAttrTrue)),
          (PylirPy_IntCmpOp (NativeCodeCall<"invertPredicate($0)"> $pred), $lhs, $rhs)>;

// py.isUnboundValue (arith.select %cond, %lhs, %rhs) and %lhs and %rhs are either both definitely bound or definitely
// not unbound -> replace with that result
def : Pat<(PylirPy_IsUnboundValueOp (SelectOp $_, $lhs, $rhs)),
            (Arith_ConstantOp (NativeCodeCall<"$_builder.getBoolAttr(*::pylir::Py::isUnbound($0))"> $lhs)),
            [(Constraint<CPred<[{
                ::pylir::Py::isUnbound($0) && ::pylir::Py::isUnbound($0) == ::pylir::Py::isUnbound($1)
            }]>> $lhs, $rhs)]>;

// py.isUnboundValue (arith.select %cond, %lhs, %rhs) and %lhs unbound and %rhs not -> %cond
def : Pat<(PylirPy_IsUnboundValueOp (SelectOp $cond, $lhs, $rhs)), (replaceWithValue $cond),
            [(Constraint<CPred<[{
                ::pylir::Py::isUnbound($0) && ::pylir::Py::isUnbound($1)
                    && *::pylir::Py::isUnbound($0) && !*::pylir::Py::isUnbound($1)
            }]>> $lhs, $rhs)]>;

// py.isUnboundValue (arith.select %cond, %lhs, %rhs) and %lhs unbound and %rhs not -> not %cond
def : Pat<(PylirPy_IsUnboundValueOp (SelectOp $cond, $lhs, $rhs)),
            (Arith_XOrIOp $cond, (Arith_ConstantOp ConstBoolAttrTrue)),
            [(Constraint<CPred<[{
                ::pylir::Py::isUnbound($0) && ::pylir::Py::isUnbound($1)
                    && !*::pylir::Py::isUnbound($0) && *::pylir::Py::isUnbound($1)
            }]>> $lhs, $rhs)]>;

multiclass IntCommAndAssocPattern<Op op, string operator> {
	def : Pat<(op (op $op, (ResolvesTo<0> PylirPy_IntAttr:$first)), (ResolvesTo<0> PylirPy_IntAttr:$second)),
			(op $op,
			(PylirPy_ConstantOp (BinOpAttrs<operator, PylirPy_IntAttr, PylirPy_IntAttr> $first, $second)))>;

	def : Pat<(op (op $x, (ResolvesTo<0> PylirPy_IntAttr:$first)), (op $y, (ResolvesTo<0> PylirPy_IntAttr:$second))),
			(op (op $x, $y),
			(PylirPy_ConstantOp (BinOpAttrs<operator, PylirPy_IntAttr, PylirPy_IntAttr> $first, $second)))>;
}

defm : IntCommAndAssocPattern<PylirPy_IntAddOp, "+">;

#endif
