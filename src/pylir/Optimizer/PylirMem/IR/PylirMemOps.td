
#ifndef PYLIR_OPS_TABLEGEN
#define PYLIR_OPS_TABLEGEN

include "pylir/Optimizer/PylirMem/IR/PylirMemTypes.td"
include "pylir/Optimizer/PylirMem/IR/PylirMemAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"

class Pylir_Op<string mneomic, list<OpTrait> traits = []> :
    Op<Pylir_Dialect, mneomic, traits>;

// Global memory operations

def Pylir_ConstantGlobalOp : Pylir_Op<"constantGlobal", [Symbol]> {

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        FlatSymbolRefAttr:$type,
        AnyAttr:$initializer
    );

    let assemblyFormat = [{
        $sym_name attr-dict `:` $type `=` $initializer
    }];

    let extraClassDeclaration = [{
        static ConstantGlobalOp create(::mlir::Location location, ::llvm::StringRef name, mlir::FlatSymbolRefAttr type, mlir::Attribute initializer);
    }];
}

def Pylir_DataOfOp : Pylir_Op<"dataOf", [NoSideEffect, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let arguments = (ins FlatSymbolRefAttr:$globalName);
    let results = (outs PointerOf<Pylir_Object>:$data);

    let assemblyFormat = [{
        $globalName attr-dict `:` type($data)
    }];

    let builders = [
        OpBuilder<(ins "pylir::Mem::ConstantGlobalOp":$value),
        [{ build($_builder, $_state, value.sym_name()); }]>];
}

def Pylir_GlobalOp : Pylir_Op<"global", [Symbol]> {
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        OptionalAttr<StrAttr>:$sym_visibility,
        TypeAttr:$type,
        OptionalAttr<AnyAttr>:$initial_value);

    let assemblyFormat = [{
           ($sym_visibility^)? $sym_name custom<GlobalInitialValue>($initial_value) attr-dict `:` $type
      }];
}

def Pylir_GetGlobalOp : Pylir_Op<"get_global", [NoSideEffect, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let arguments = (ins FlatSymbolRefAttr:$name);
    let results = (outs AnyPointer:$result);

    let assemblyFormat = "$name attr-dict `:` type($result)";
}

// Memory allocation

def Pylir_GCObjectAllocOp : Pylir_Op<"gcObjectAlloc">
{
    let arguments = (ins Optional<Index>:$variableSize, FlatSymbolRefAttr:$type);
    let results = (outs Res<PointerOf<Pylir_Object>, "Pointer to allocated object", [MemAlloc]>:$pointer);

    let assemblyFormat = [{
        $type (`[` $variableSize^ `]`)? attr-dict `:` type($pointer)
    }];

    let verifier = [{
        return ::verifyDynamicSize(*this);
    }];
}

def Pylir_ObjectAllocaOp : Pylir_Op<"objectAlloca">
{
    let arguments = (ins Optional<Index>:$variableSize, FlatSymbolRefAttr:$type);
    let results = (outs Res<PointerOf<Pylir_Object>, "Pointer to allocated object", [MemAlloc]>:$pointer);

    let assemblyFormat = [{
        $type (`[` $variableSize^ `]`)? attr-dict `:` type($pointer)
    }];

    let verifier = [{
        return ::verifyDynamicSize(*this);
    }];
}

def Pylir_AllocaOp : Pylir_Op<"alloca"> {
    let arguments = (ins );
    let results = (outs Res<AnyPointer, "the allocated memory", [MemAlloc<AutomaticAllocationScopeResource>]>:$pointer);

    let assemblyFormat = "attr-dict `:` type($pointer)";
}

// Memory access Operation

def Pylir_StoreOp : Pylir_Op<"store", [TypesMatchWith<"value type matches element type of 'pointer'",
                                                      "pointer", "value",
                                                      "$_self.cast<::pylir::Mem::PointerType>().getElementType()">
                                                      ]> {
    let summary = "store operation";

    let arguments = (ins AnyType:$value, Arg<AnyPointer, "the pointer to store to", [MemWrite]>:$pointer);

    let assemblyFormat = [{
        $value `:` type($value) `,` $pointer attr-dict `:` type($pointer)
      }];
}

def Pylir_LoadOp : Pylir_Op<"load", [TypesMatchWith<"result type matches element type of 'pointer'",
                                                    "pointer", "result",
                                                    "$_self.cast<pylir::Mem::PointerType>().getElementType()">]> {
    let summary = "load operation";

    let arguments = (ins Arg<AnyPointer, "the pointer to load from", [MemRead]>:$pointer);
    let results = (outs AnyType:$result);

    let assemblyFormat = "$pointer attr-dict `:` type($pointer)";

    let builders = [
        OpBuilder<(ins "mlir::Value":$pointer),[{
            return build($_builder,$_state, pointer.getType().cast<pylir::Mem::PointerType>().getElementType(), pointer);
        }]>
    ];
}

// Value type I/O

def Pylir_BoxIntoOp : Pylir_Op<"boxInto"> {
    let arguments = (ins AnyValueType:$input, Arg<PointerOf<Pylir_Object>, "Pointer to object to store into", [MemWrite]>:$pointer);
    let results = (outs );

    let assemblyFormat = [{
        $input $pointer attr-dict `:` type($input) `,` type($pointer)
    }];
}

def Pylir_UnboxOp : Pylir_Op<"unbox"> {
    let arguments = (ins Arg<PointerOf<Pylir_Object>, "Pointer to object to read from", [MemRead]>:$input);
    let results = (outs AnyValueType:$result);

    let assemblyFormat = [{
        $input attr-dict `:` type($input) `->` type($result)
    }];
}

// Object Ops

def Pylir_TypeOfOp : Pylir_Op<"typeOf", [NoSideEffect]> {
    let arguments = (ins PointerOf<Pylir_Object>:$input);
    let results = (outs PointerOf<Pylir_Object>:$result);

    let assemblyFormat = [{
                  $input attr-dict `:` type($input) `->` type($result)
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
def Pylir_TypeSlot_MRO : I32EnumAttrCase<"Mro", 64, "__mro__">;
def Pylir_TypeSlot_Name : I32EnumAttrCase<"Name", 65, "__name__">;

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
    Pylir_TypeSlot_New, Pylir_TypeSlot_Bases, Pylir_TypeSlot_Del, Pylir_TypeSlot_MRO, Pylir_TypeSlot_Name
]> {
    let cppNamespace = "::pylir::Mem";
}

def Pylir_GetTypeSlotOp : Pylir_Op<"getTypeSlot", [DeclareOpInterfaceMethods<InferTypeOpInterface>, NoSideEffect]> {
    let arguments = (ins Pylir_TypeSlotPredicateAttr:$predicate, Arg<PointerOf<Pylir_Object>, "", [MemRead]>:$input);
    let results = (outs AnyType:$result, I1:$found);

    let assemblyFormat = "$predicate `from` $input attr-dict `:` type($input) `->` type($result) `,` type($found)";

    let extraClassDeclaration = [{
        static mlir::Type returnTypeFromPredicate(mlir::MLIRContext* context,TypeSlotPredicate predicate);
    }];
}

def Pylir_IdOp : Pylir_Op<"id", [NoSideEffect]> {
    let arguments = (ins PointerOf<Pylir_Object>:$input);
    let results = (outs Index:$result);

    let assemblyFormat = [{
                      $input attr-dict `:` type($input) `->` type($result)
                    }];
}

// Tuple Ops

def Pylir_GetTupleItemOp : Pylir_Op<"getTupleItem", [NoSideEffect]> {
    let arguments = (ins PointerOf<Pylir_Object>:$tuple, Index:$index);
    let results = (outs PointerOf<Pylir_Object>:$result);

    let assemblyFormat = [{
        $tuple `[` $index `]` attr-dict `:` type($tuple) `,` type($index) `->` type($result)
    }];
}

def Pylir_SetTupleItemOp : Pylir_Op<"setTupleItem"> {
    let summary = "create a tuple";

    let arguments = (ins
                        Arg<PointerOf<Pylir_Object>, "Tuple to write to",[MemWrite]>:$memory,
                        Index:$index,
                        PointerOf<Pylir_Object>:$element);

    let assemblyFormat = [{
        $memory `[` $index `]` `to` $element attr-dict `:` type($memory) `,` type($index) `->` type($element)
    }];
}

def Pylir_TupleSizeOp : Pylir_Op<"tupleSize"> {
    let summary = "get tuple size";

    let arguments = (ins
                        Arg<PointerOf<Pylir_Object>, "Tuple to read from", [MemRead]>:$tuple);

    let results = (outs Index:$size);

    let assemblyFormat = [{
        $tuple attr-dict
    }];
}

// Integer ops

def Pylir_IntegerConstant : Pylir_Op<"intConst"> {
    let arguments = (ins Arg<PointerOf<Pylir_Object>, "TypeObject", [MemRead]>:$typeObject, Pylir_IntegerAttr:$value);
    let results = (outs Arg<PointerOf<Pylir_Object>, "Result", [MemAlloc]>:$result);

    let assemblyFormat = [{
        $value `of` $typeObject attr-dict `:` type($typeObject) `->` type($result)
    }];
}

class Pylir_IBinOp<string name, list<OpTrait> traits = []> : Pylir_Op<name,traits> {
    let arguments = (ins
                        Arg<PointerOf<Pylir_Object>, "Left operand", [MemRead]>:$lhs,
                        Arg<PointerOf<Pylir_Object>, "Right operand", [MemRead]>:$rhs);
    let results = (outs Arg<PointerOf<Pylir_Object>, "Result", [MemAlloc]>:$result);

    let assemblyFormat = [{
          $lhs `,` $rhs  attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
    }];
}

def Pylir_IAddOp : Pylir_IBinOp<"iadd", [Commutative]>;

def Pylir_ISubOp : Pylir_IBinOp<"isub">;

def Pylir_IMulOp : Pylir_IBinOp<"imul", [Commutative]>;

def Pylir_IDivOp : Pylir_IBinOp<"itruediv"> {
    let results = (outs Pylir_Float:$result);
}

def Pylir_IFloorDivOp : Pylir_IBinOp<"ifloordiv">;

def Pylir_IModOp : Pylir_IBinOp<"imod">;

def Pylir_INegOp : Pylir_Op<"ineg", [Involution, SameOperandsAndResultType]> {

    let arguments = (ins Arg<PointerOf<Pylir_Object>, "Operand", [MemRead]>:$input);
    let results = (outs Arg<PointerOf<Pylir_Object>, "Result", [MemAlloc]>:$result);

    let assemblyFormat = [{
        $input attr-dict `:` type($input) `->` type($result)
    }];
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
    let cppNamespace = "::pylir::Mem";
}

def Pylir_ICmpOp : Pylir_Op<"icmp", [SameTypeOperands, NoSideEffect]> {
    let arguments = (ins Pylir_CmpPredicateAttr:$predicate,
                         Arg<PointerOf<Pylir_Object>, "Left operand", [MemRead]>:$lhs,
                         Arg<PointerOf<Pylir_Object>, "Right operand", [MemRead]>:$rhs);
    let results = (outs Pylir_Bool:$result);

    let assemblyFormat = "$predicate $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Pylir_ItoFOp : Pylir_Op<"iTof"> {
    let arguments = (ins Arg<PointerOf<Pylir_Object>, "Left operand", [MemRead]>:$input);
    let results = (outs Pylir_Float:$result);

    let assemblyFormat = [{
                  $input attr-dict `:` type($input) `->` type($result)
    }];
}

def Pylir_ItoIndexOp : Pylir_Op<"iToIndex"> {
    let arguments = (ins Arg<PointerOf<Pylir_Object>, "Left operand", [MemRead]>:$input);
    let results = (outs Index:$result, I1:$overflow);

    let assemblyFormat = [{
              $input attr-dict `:` type($input) `->` type($result) `,` type($overflow)
    }];
}

// String ops

def Pylir_GetStringItemOp : Pylir_Op<"getStringItem", [NoSideEffect]> {
    let arguments = (ins PointerOf<Pylir_Object>:$string, Index:$index);
    let results = (outs I32:$result);

    let assemblyFormat = [{
        $string `[` $index `]` attr-dict `:` type($string) `,` type($index) `->` type($result)
    }];
}

def Pylir_SetStringItemOp : Pylir_Op<"setStringItem"> {
    let summary = "set a string item";

    let arguments = (ins
                        Arg<PointerOf<Pylir_Object>, "String to write to", [MemWrite]>:$string,
                        Index:$index,
                        I32:$element);

    let assemblyFormat = [{
        $string `[` $index `]` `to` $element attr-dict `:` type($string) `,` type($index) `->` type($element)
    }];
}

def Pylir_StringSizeOp : Pylir_Op<"stringSize"> {
    let summary = "get string size";

    let arguments = (ins
                        Arg<PointerOf<Pylir_Object>, "String to read from", [MemRead]>:$string);

    let results = (outs Index:$size);

    let assemblyFormat = [{
        $string attr-dict
    }];
}

def Pylir_RaiseOp : Pylir_Op<"raise", [Terminator]> {
    let arguments = (ins PointerOf<Pylir_Object>:$exception);
    let results = (outs);

    let assemblyFormat = "$exception `:` type($exception) attr-dict";
}

#endif