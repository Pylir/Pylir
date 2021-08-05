
#ifndef PYLIR_OPS_TABLEGEN
#define PYLIR_OPS_TABLEGEN

include "pylir/Optimizer/Dialect/PylirTypes.td"
include "pylir/Optimizer/Dialect/PylirAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"

class Pylir_Op<string mneomic, list<OpTrait> traits = []> :
    Op<Pylir_Dialect, mneomic, traits>;

def Pylir_ConstantGlobalOp : Pylir_Op<"constantGlobal", [Symbol, NoSideEffect]> {

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        Pylir_KnownTypeObjectAttr:$type,
        AnyAttr:$initializer
    );

    let assemblyFormat = [{
        $sym_name `->` $type attr-dict `=` $initializer
    }];

    let extraClassDeclaration = [{
        static ConstantGlobalOp create(::mlir::Location location, ::llvm::StringRef name, ObjectType type, mlir::Attribute initializer);
    }];
}

def Pylir_DataOfOp : Pylir_Op<"dataOf",[NoSideEffect, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let arguments = (ins FlatSymbolRefAttr:$globalName);
    let results = (outs PointerOf<Pylir_Object>:$data);

    let assemblyFormat = [{
        $globalName `->` type($data) attr-dict
    }];

    let builders = [
        OpBuilder<(ins "pylir::Dialect::ConstantGlobalOp":$value),
        [{ build($_builder, $_state, ::pylir::Dialect::PointerType::get(value.type()), value.sym_name()); }]>];
}

def Pylir_ConstantOp : Pylir_Op<"constant",
    [ConstantLike, NoSideEffect, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let arguments = (ins AnyAttr:$value);
    let results = (outs AnyValueType:$result);

    let assemblyFormat = [{
            $value `->` type($result) attr-dict
        }];

    let hasFolder = 1;
}

class Pylir_BinOp<string name, list<OpTrait> traits = []> : Pylir_Op<name,traits> {
    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
          $lhs `:` type($lhs) `,` $rhs `:` type($rhs) `->` type($result) attr-dict
        }];

    let hasFolder = 1;
}

class Pylir_IBinOp<string name, list<OpTrait> traits = []>
    : Pylir_BinOp<name,!listconcat(traits, [NoSideEffect, SameOperandsAndResultType])> {
    let arguments = (ins Pylir_Integer:$lhs, Pylir_Integer:$rhs);
    let results = (outs Pylir_Integer:$result);
}

def Pylir_IAddOp : Pylir_IBinOp<"iadd", [Commutative]>;

def Pylir_ISubOp : Pylir_IBinOp<"isub">;

def Pylir_IMulOp : Pylir_IBinOp<"imul", [Commutative]>;

def Pylir_IDivOp : Pylir_BinOp<"itruediv", [SameTypeOperands]> {
    let results = (outs Pylir_Float:$result);
}

def Pylir_IFloorDivOp : Pylir_IBinOp<"ifloordiv">;

def Pylir_IModOp : Pylir_IBinOp<"imod">;

def Pylir_INegOp : Pylir_Op<"ineg",[SameOperandsAndResultType, Involution]> {

    let arguments = (ins Pylir_Integer:$input);
    let results = (outs Pylir_Integer:$result);

    let assemblyFormat = [{
        $input `:` type($input) `->` type($result) attr-dict
    }];

    let hasFolder = 1;
}

def Pylir_IShlOp : Pylir_IBinOp<"ishl">;

def Pylir_IShrOp : Pylir_IBinOp<"ishr">;

def Pylir_IAndOp : Pylir_IBinOp<"iand", [Commutative]>;

def Pylir_IXorOp : Pylir_IBinOp<"ixor", [Commutative]>;

def Pylir_IOrOp : Pylir_IBinOp<"ior", [Commutative]>;

def Pylir_CmpEQ : I32EnumAttrCase<"EQ", 0, "eq">;
def Pylir_CmpNE : I32EnumAttrCase<"NE", 1, "ne">;
def Pylir_CmpLT : I32EnumAttrCase<"LT", 2, "lt">;
def Pylir_CmpLE : I32EnumAttrCase<"LE", 3, "le">;
def Pylir_CmpGT : I32EnumAttrCase<"GT", 4, "gt">;
def Pylir_CmpGE : I32EnumAttrCase<"GE", 5, "ge">;

def Pylir_CmpPredicateAttr : I32EnumAttr<"CmpPredicate","",[
    Pylir_CmpEQ, Pylir_CmpNE, Pylir_CmpLT, Pylir_CmpLE, Pylir_CmpGT, Pylir_CmpGE
]> {
    let cppNamespace = "::pylir::Dialect";
}

def Pylir_ICmpOp : Pylir_Op<"icmp", [SameTypeOperands, NoSideEffect]> {
    let arguments = (ins Pylir_CmpPredicateAttr:$predicate, Pylir_Integer:$lhs, Pylir_Integer:$rhs);
    let results = (outs Pylir_Bool:$result);

    let assemblyFormat = "$predicate $lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict";

    let hasFolder = 1;
}

class Pylir_Cast<string name, list<OpTrait> traits = []>
    : Pylir_Op<name,!listconcat(traits,[NoSideEffect, DeclareOpInterfaceMethods<CastOpInterface>])> {
    let assemblyFormat = [{
          $input `:` type($input) `->` type($result) attr-dict
        }];
}

def Pylir_ItoFOp : Pylir_Cast<"iTof"> {
    let arguments = (ins Pylir_Integer:$input);
    let results = (outs Pylir_Float:$result);

    let hasFolder = 1;
}

def Pylir_ItoIndexOp : Pylir_Op<"iToIndex", [NoSideEffect]> {
    let arguments = (ins Pylir_Integer:$input);
    let results = (outs Index:$result, I1:$overflow);

    let assemblyFormat = [{
              $input attr-dict
    }];

    let hasFolder = 1;
}

def Pylir_BtoIOp : Pylir_Cast<"bToi"> {
    let arguments = (ins Pylir_Bool:$input);
    let results = (outs Pylir_Integer:$result);

    let hasFolder = 1;
}

def Pylir_MakeListOp : Pylir_Op<"makeList", [NoSideEffect]> {
    let arguments = (ins Variadic<PointerOf<Pylir_Object>>:$input);
    let results = (outs Pylir_List:$result);

    let assemblyFormat = [{
              $input `:` type($input) `->` type($result) attr-dict
            }];

    let hasFolder = 1;
}

def Pylir_MakeTupleOp : Pylir_Op<"makeTuple", [NoSideEffect]> {
    let arguments = (ins Variadic<PointerOf<Pylir_Object>>:$input);
    let results = (outs Pylir_Tuple:$result);

    let assemblyFormat = [{
              $input `:` type($input) `->` type($result) attr-dict
            }];

    let hasFolder = 1;
}

def Pylir_TupleToListOp : Pylir_Cast<"tupleToList", [NoSideEffect]> {
    let arguments = (ins Pylir_Tuple:$input);
    let results = (outs Pylir_List:$result);

    let hasFolder = 1;
}

def Pylir_ListToTupleOp : Pylir_Cast<"listToTuple", [NoSideEffect]> {
    let arguments = (ins Pylir_List:$input);
    let results = (outs Pylir_Tuple:$result);

    let hasFolder = 1;
}

def Pylir_TypeOfOp : Pylir_Op<"typeOf", [NoSideEffect]> {
    let arguments = (ins Pylir_Object:$input);
    let results = (outs PointerOf<Pylir_TypeObject>:$result);

    let assemblyFormat = [{
                  $input `:` type($input) `->` type($result) attr-dict
                }];
}

def Pylir_TypeSlot_DictPtr : I32EnumAttrCase<"DictPtr", 1, "__dictoffset__">;
def Pylir_TypeSlot_Call : I32EnumAttrCase<"Call", 2, "__call__">;
def Pylir_TypeSlot_Add : I32EnumAttrCase<"Add", 3, "__add__">;
def Pylir_TypeSlot_Subtract : I32EnumAttrCase<"Subtract", 4, "__sub__">;
def Pylir_TypeSlot_Multiply : I32EnumAttrCase<"Multiply", 5, "__mul__">;
def Pylir_TypeSlot_Remainder : I32EnumAttrCase<"Remainder", 6, "__rem__">;
def Pylir_TypeSlot_DivMod : I32EnumAttrCase<"Divmod", 7, "__divmod__">;
def Pylir_TypeSlot_Power : I32EnumAttrCase<"Power", 8, "__pow__">;
def Pylir_TypeSlot_Negative : I32EnumAttrCase<"Negative", 9, "__neg__">;
def Pylir_TypeSlot_Positive : I32EnumAttrCase<"Positive", 10, "__pos__">;
def Pylir_TypeSlot_Absolute : I32EnumAttrCase<"Absolute", 11, "__abs__">;
def Pylir_TypeSlot_Bool : I32EnumAttrCase<"Bool", 12, "__bool__">;
def Pylir_TypeSlot_Invert : I32EnumAttrCase<"Invert", 13, "__inv__">;
def Pylir_TypeSlot_LShift : I32EnumAttrCase<"LShift", 14, "__lshift__">;
def Pylir_TypeSlot_RShift : I32EnumAttrCase<"RShift", 15, "__rshift__">;
def Pylir_TypeSlot_And : I32EnumAttrCase<"And", 16, "__and__">;
def Pylir_TypeSlot_Xor : I32EnumAttrCase<"Xor", 17, "__xor__">;
def Pylir_TypeSlot_Or : I32EnumAttrCase<"Or", 18, "__or__">;
def Pylir_TypeSlot_Int : I32EnumAttrCase<"Int", 19, "__int__">;
def Pylir_TypeSlot_Float : I32EnumAttrCase<"Float", 20, "__float__">;
def Pylir_TypeSlot_InPlace_Add : I32EnumAttrCase<"InPlaceAdd", 21, "__iadd__">;
def Pylir_TypeSlot_InPlace_Subtract : I32EnumAttrCase<"InPlaceSubtract", 22, "__isub__">;
def Pylir_TypeSlot_InPlace_Multiply : I32EnumAttrCase<"InPlaceMultiply", 23, "__imul__">;
def Pylir_TypeSlot_InPlace_Remainder : I32EnumAttrCase<"InPlaceRemainder", 24, "__irem__">;
def Pylir_TypeSlot_InPlace_Power : I32EnumAttrCase<"InPlacePower", 25, "__ipow__">;
def Pylir_TypeSlot_InPlace_LShift : I32EnumAttrCase<"InPlaceLShift", 26, "__ilshift__">;
def Pylir_TypeSlot_InPlace_RShift : I32EnumAttrCase<"InPlaceRShift", 27, "__irshift__">;
def Pylir_TypeSlot_InPlace_And : I32EnumAttrCase<"InPlaceAnd", 28, "__iand__">;
def Pylir_TypeSlot_InPlace_Xor : I32EnumAttrCase<"InPlaceXor", 29, "__ixor__">;
def Pylir_TypeSlot_InPlace_Or : I32EnumAttrCase<"InPlaceOr", 30, "__ior__">;
def Pylir_TypeSlot_FloorDivide : I32EnumAttrCase<"FloorDivide", 31, "__floordiv__">;
def Pylir_TypeSlot_TrueDivide : I32EnumAttrCase<"TrueDivide", 32, "__truediv__">;
def Pylir_TypeSlot_InPlace_FloorDivide : I32EnumAttrCase<"InPlaceFloorDivide", 33, "__ifloordiv__">;
def Pylir_TypeSlot_InPlace_TrueDivide : I32EnumAttrCase<"InPlaceTrueDivide", 34, "__itruediv__">;
def Pylir_TypeSlot_Index : I32EnumAttrCase<"Index", 35, "__index__">;
def Pylir_TypeSlot_MatrixMultiply : I32EnumAttrCase<"MatrixMultiply", 36, "__matmul__">;
def Pylir_TypeSlot_InPlace_MatrixMultiply : I32EnumAttrCase<"InPlaceMatrixMultiply", 37, "__imatmul__">;
def Pylir_TypeSlot_Length : I32EnumAttrCase<"Length", 38, "__len__">;
def Pylir_TypeSlot_GetItem : I32EnumAttrCase<"GetItem", 39, "__getitem__">;
def Pylir_TypeSlot_SetItem : I32EnumAttrCase<"SetItem", 40, "__setitem__">;
def Pylir_TypeSlot_DelItem : I32EnumAttrCase<"DelItem", 41, "__delitem__">;
def Pylir_TypeSlot_Missing : I32EnumAttrCase<"Missing", 42, "__missing__">;
def Pylir_TypeSlot_Iter : I32EnumAttrCase<"Iter", 43, "__iter__">;
def Pylir_TypeSlot_Contains : I32EnumAttrCase<"Contains", 44, "__contains__">;
def Pylir_TypeSlot_Hash : I32EnumAttrCase<"Hash", 45, "__hash__">;
def Pylir_TypeSlot_Str : I32EnumAttrCase<"Str", 46, "__str__">;
def Pylir_TypeSlot_Repr : I32EnumAttrCase<"Repr", 47, "__repr__">;
def Pylir_TypeSlot_GetAttr : I32EnumAttrCase<"GetAttr", 48, "__getattr__">;
def Pylir_TypeSlot_SetAttr : I32EnumAttrCase<"SetAttr", 49, "__setattr__">;
def Pylir_TypeSlot_Eq : I32EnumAttrCase<"Eq", 50, "__eq__">;
def Pylir_TypeSlot_Ne : I32EnumAttrCase<"Ne", 51, "__ne__">;
def Pylir_TypeSlot_Lt : I32EnumAttrCase<"Lt", 52, "__lt__">;
def Pylir_TypeSlot_Gt : I32EnumAttrCase<"Gt", 53, "__gt__">;
def Pylir_TypeSlot_Le : I32EnumAttrCase<"Le", 54, "__le__">;
def Pylir_TypeSlot_Ge : I32EnumAttrCase<"Ge", 55, "__ge__">;
def Pylir_TypeSlot_IterNext : I32EnumAttrCase<"IterNext", 56, "__next__">;
def Pylir_TypeSlot_Dict : I32EnumAttrCase<"Dict", 57, "__dict__">;
def Pylir_TypeSlot_DescrGet : I32EnumAttrCase<"DescrGet", 58, "__get__">;
def Pylir_TypeSlot_DescrSet : I32EnumAttrCase<"DescrSet", 59, "__set__">;
def Pylir_TypeSlot_Init : I32EnumAttrCase<"Init", 60, "__init__">;
def Pylir_TypeSlot_New : I32EnumAttrCase<"New", 61, "__new__">;
def Pylir_TypeSlot_Bases : I32EnumAttrCase<"Bases", 62, "__bases__">;
def Pylir_TypeSlot_Del : I32EnumAttrCase<"Del", 63, "__del__">;

def Pylir_TypeSlotPredicateAttr : I32EnumAttr<"TypeSlotPredicate","",[
    Pylir_TypeSlot_DictPtr, Pylir_TypeSlot_Call, Pylir_TypeSlot_Add, Pylir_TypeSlot_Subtract, Pylir_TypeSlot_Multiply,
    Pylir_TypeSlot_Remainder, Pylir_TypeSlot_DivMod, Pylir_TypeSlot_Power, Pylir_TypeSlot_Negative,
    Pylir_TypeSlot_Positive, Pylir_TypeSlot_Absolute, Pylir_TypeSlot_Bool, Pylir_TypeSlot_Invert, Pylir_TypeSlot_LShift,
    Pylir_TypeSlot_RShift, Pylir_TypeSlot_And, Pylir_TypeSlot_Xor, Pylir_TypeSlot_Or, Pylir_TypeSlot_Int,
    Pylir_TypeSlot_Float, Pylir_TypeSlot_InPlace_Add, Pylir_TypeSlot_InPlace_Subtract, Pylir_TypeSlot_InPlace_Multiply,
    Pylir_TypeSlot_InPlace_Remainder, Pylir_TypeSlot_InPlace_Power, Pylir_TypeSlot_InPlace_LShift,
    Pylir_TypeSlot_InPlace_RShift, Pylir_TypeSlot_InPlace_And, Pylir_TypeSlot_InPlace_Xor, Pylir_TypeSlot_InPlace_Or,
    Pylir_TypeSlot_FloorDivide, Pylir_TypeSlot_TrueDivide, Pylir_TypeSlot_InPlace_FloorDivide,
    Pylir_TypeSlot_InPlace_TrueDivide, Pylir_TypeSlot_Index, Pylir_TypeSlot_MatrixMultiply,
    Pylir_TypeSlot_InPlace_MatrixMultiply, Pylir_TypeSlot_Length, Pylir_TypeSlot_GetItem, Pylir_TypeSlot_SetItem,
    Pylir_TypeSlot_DelItem, Pylir_TypeSlot_Missing, Pylir_TypeSlot_Iter, Pylir_TypeSlot_Contains, Pylir_TypeSlot_Hash,
    Pylir_TypeSlot_Str, Pylir_TypeSlot_Repr, Pylir_TypeSlot_GetAttr, Pylir_TypeSlot_SetAttr, Pylir_TypeSlot_Eq,
    Pylir_TypeSlot_Ne, Pylir_TypeSlot_Lt, Pylir_TypeSlot_Gt, Pylir_TypeSlot_Le, Pylir_TypeSlot_Ge,
    Pylir_TypeSlot_IterNext, Pylir_TypeSlot_Dict, Pylir_TypeSlot_DescrGet, Pylir_TypeSlot_DescrSet, Pylir_TypeSlot_Init,
    Pylir_TypeSlot_New, Pylir_TypeSlot_Bases, Pylir_TypeSlot_Del
]> {
    let cppNamespace = "::pylir::Dialect";
}

def Pylir_GetTypeSlotOp : Pylir_Op<"getTypeSlot", [DeclareOpInterfaceMethods<InferTypeOpInterface>, NoSideEffect]> {
    let arguments = (ins Pylir_TypeSlotPredicateAttr:$predicate, Arg<Pylir_TypeObject,"",[MemRead]>:$input);
    let results = (outs AnyType:$result, I1:$found);

    let assemblyFormat = "$predicate `from` $input `:` type($input) `->` type($result) attr-dict";

    let extraClassDeclaration = [{
        static mlir::Type returnTypeFromPredicate(mlir::MLIRContext* context,TypeSlotPredicate predicate);
    }];
}

def Pylir_IdOp : Pylir_Op<"id", [NoSideEffect]> {
    let arguments = (ins PointerOf<Pylir_Object>:$input);
    let results = (outs Index:$result);

    let assemblyFormat = [{
                      $input `:` type($input) attr-dict
                    }];
}

def Pylir_RaiseOp : Pylir_Op<"raise", [Terminator]> {
    let arguments = (ins PointerOf<Pylir_Object>:$exception);
    let results = (outs);

    let assemblyFormat = "$exception `:` type($exception) attr-dict";
}

def Pylir_BoxOp : Pylir_Cast<"box", [NoSideEffect]> {
    let arguments = (ins AnyValueType:$input);
    let results = (outs Pylir_KnownTypeObject:$result);

    let assemblyFormat = [{
        $input `:` type($input) `->` type($result) attr-dict
    }];
}

def Pylir_UnboxOp : Pylir_Cast<"unbox", [NoSideEffect]> {
    let arguments = (ins PointerOf<Pylir_Object>:$input);
    let results = (outs AnyImmutableType:$result);

    let assemblyFormat = [{
        $input `:` type($input) `->` type($result) attr-dict
    }];
}

def Pylir_GCAllocOp : Pylir_Op<"gcAlloc">
{
    let arguments = (ins Optional<Index>:$dynamicSize);
    let results = (outs Res<NotUnknownObjectPointer, "Pointer to allocated object", [MemAlloc]>:$pointer);

    let assemblyFormat = [{
        type($pointer) (`[` $dynamicSize^ `]`)? attr-dict
    }];

    let builders = [
                OpBuilder<(ins "mlir::Type":$pointer),[{
                    return build($_builder,$_state, pointer, {});
                }]>
            ];
}

def Pylir_GetRefOp : Pylir_Op<"getRef", [NoSideEffect]>
{
    let arguments = (ins PointerOf<Pylir_Object>:$input);
    let results = (outs PointerOf<AnyMutableType>:$result);

    let assemblyFormat = [{
            $input `:` type($input) `->` type($result) attr-dict
        }];
}

def Pylir_GetTupleItemOp : Pylir_Op<"getTupleItem", [NoSideEffect]> {
    let arguments = (ins Pylir_Tuple:$tuple, Index:$index);
    let results = (outs PointerOf<Pylir_Object>:$result);

    let assemblyFormat = [{
        $tuple `[` $index `]` `->` type($result) attr-dict
    }];
}

def Pylir_GetStringItemOp : Pylir_Op<"getStringItem", [NoSideEffect]> {
    let arguments = (ins Pylir_String:$string, Index:$index);
    let results = (outs Pylir_String:$result);

    let assemblyFormat = [{
        $string `[` $index `]` attr-dict
    }];

    let hasFolder = 1;
}

def Pylir_GetListItemOp : Pylir_Op<"getListItem", [NoSideEffect]> {
    let arguments = (ins PointerOf<Pylir_List>:$list, Index:$index);
    let results = (outs PointerOf<Pylir_Object>:$result);

    let assemblyFormat = [{
        $list `[` $index `]` `->` type($result) attr-dict
    }];
}

def Pylir_CallOp : Pylir_Op<"call",
    [DeclareOpInterfaceMethods<CallOpInterface>,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {

    let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
    let results = (outs Variadic<AnyType>);

    let summary = "call operation";
    let description = [{
        Same as std.call, but allows implicit conversion from pylir.object<*> to pylir.object.
        In the future it'll almost certainly be used to allow exception handling
    }];

    let assemblyFormat = [{
        $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
    }];
    let verifier = ?;
}

def Pylir_CallIndirectOp : Pylir_Op<"call_indirect",[DeclareOpInterfaceMethods<CallOpInterface>,
                                                 TypesMatchWith<"callee result types match result types",
                                                 "callee", "results",
                                                 "$_self.cast<FunctionType>().getResults()">]> {
    let summary = "indirect call operation";
    let description = [{
        Same as std.call_indirect, but allows implicit conversion from pylir.object<*> to pylir.object.
        In the future it'll almost certainly be used to allow exception handling
    }];

    let arguments = (ins FunctionType:$callee, Variadic<AnyType>:$operands);
    let results = (outs Variadic<AnyType>:$results);

    let verifier = ?;

    let builders = [
        OpBuilder<(ins "mlir::Value":$callee, CArg<"mlir::ValueRange", "{}">:$operands), [{
            return build($_builder, $_state, callee.getType().cast<mlir::FunctionType>().getResults(), callee, operands);
        }]>
    ];

    let assemblyFormat = "$callee `:` type($callee) `(` $operands `)` attr-dict `:` functional-type($operands, results)";
}

def Pylir_ReturnOp : Pylir_Op<"return",[NoSideEffect, HasParent<"::mlir::FuncOp">,
                                        ReturnLike, Terminator]> {
    let summary = "return operation";
    let description = "Same as std.return but allows implicit conversion from pylir.object<*> to pylir.boject";

    let arguments = (ins Variadic<AnyType>:$operands);
    let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def Pylir_StoreOp : Pylir_Op<"store"> {
    let summary = "store operation";
    let description = "Same as memref.store but allows implicit conversion from pylir.object<*> to pylir.boject";

    let arguments = (ins AnyType:$value,
                         Arg<AnyPointer, "the pointer to store to",[MemWrite]>:$pointer);

    let assemblyFormat = [{
        $value `:` type($value) `,` $pointer attr-dict `:` type($pointer)
      }];
}

def Pylir_LoadOp : Pylir_Op<"load", [TypesMatchWith<"result type matches element type of 'pointer'",
                                                    "pointer", "result",
                                                    "$_self.cast<pylir::Dialect::PointerType>().getElementType()">]> {
    let summary = "load operation";

    let arguments = (ins Arg<AnyPointer, "the pointer to load from",
                               [MemRead]>:$pointer);
    let results = (outs AnyType:$result);

    let assemblyFormat = "$pointer attr-dict `:` type($pointer)";

    let builders = [
        OpBuilder<(ins "mlir::Value":$pointer),[{
            return build($_builder,$_state, pointer.getType().cast<pylir::Dialect::PointerType>().getElementType(), pointer);
        }]>
    ];
}

def Pylir_AllocaOp : Pylir_Op<"alloca"> {
    let arguments = (ins Optional<Index>:$dynamicSize);
    let results = (outs Res<AnyPointer, "the allocated memory", [MemAlloc<AutomaticAllocationScopeResource>]>:$pointer);

    let assemblyFormat = "(`[` $dynamicSize^ `]`)? attr-dict `:` type($pointer)";

    let builders = [
            OpBuilder<(ins "mlir::Type":$pointer),[{
                return build($_builder,$_state, pointer, {});
            }]>
        ];
}

def Pylir_GlobalOp : Pylir_Op<"global", [Symbol]> {
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        OptionalAttr<StrAttr>:$sym_visibility,
        TypeAttr:$type,
        OptionalAttr<AnyAttr>:$initial_value);

    let assemblyFormat = [{
           ($sym_visibility^)? $sym_name `:` $type custom<GlobalInitialValue>($initial_value) attr-dict
      }];
}

def Pylir_GetGlobalOp : Pylir_Op<"get_global", [NoSideEffect, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let arguments = (ins FlatSymbolRefAttr:$name);
    let results = (outs AnyPointer:$result);

    let assemblyFormat = "$name `:` type($result) attr-dict";
}

#endif
