
#ifndef PYLIR_OPS_TABLEGEN
#define PYLIR_OPS_TABLEGEN

include "PylirTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"

class Pylir_Op<string mneomic, list<OpTrait> traits = []> :
    Op<Pylir_Dialect, mneomic, traits> {
    let hasFolder = 1;
}

def Pylir_GlobalOp : Pylir_Op<"global", [Symbol, NoSideEffect]> {
    let hasFolder = 0;

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttr:$type,
        UnitAttr:$constant,
        OptionalAttr<Pylir_DictAttr>:$dict
    );

    let assemblyFormat = [{
             (`constant` $constant^)? $sym_name `->` $type attr-dict
        }];

    let extraClassDeclaration = [{
        static GlobalOp create(::mlir::Location location, ::llvm::StringRef name, mlir::Type type, bool constant = false, ::pylir::Dialect::DictAttr initializer = {});

        static GlobalOp create(::mlir::Location location, ::llvm::StringRef name, bool constant = false, ::pylir::Dialect::DictAttr initializer = {});
    }];
}

def Pylir_HandleOfOp : Pylir_Op<"handleof",[NoSideEffect, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let arguments = (ins FlatSymbolRefAttr:$globalName);
    let results = (outs Pylir_Handle:$handle);

    let hasFolder = 0;

    let assemblyFormat = [{
        $globalName attr-dict
    }];
}

def Pylir_AllocaOp : Pylir_Op<"alloca"> {
    let hasFolder = 0;

    let results = (outs Res<Pylir_Handle, "", [MemAlloc<AutomaticAllocationScopeResource>]>:$handle);

    let assemblyFormat = [{
         `->` type($handle) attr-dict
    }];
}

def Pylir_LoadOp : Pylir_Op<"load"> {
    let hasFolder = 0;

    let arguments = (ins Arg<Pylir_Handle, "", [MemRead]>:$handle);
    let results = (outs AnyPythonType:$result);

    let assemblyFormat = [{
        $handle `->` type($result) attr-dict
    }];
}

def Pylir_StoreOp : Pylir_Op<"store"> {
    let hasFolder = 0;

    let arguments = (ins AnyPythonType:$value, Arg<Pylir_Handle, "", [MemWrite]>:$handle);

    let assemblyFormat = [{
           $value `:` type($value) `,` $handle attr-dict
    }];
}

def Pylir_ConstantOp : Pylir_Op<"constant",[ConstantLike, NoSideEffect]> {
    let arguments = (ins AnyAttr:$value);
    let results = (outs AnyType:$result);

    let builders = [
        OpBuilderDAG<(ins "mlir::Attribute":$value),
        [{ build($_builder, $_state, value.getType(), value); }]>];

    let assemblyFormat = [{
            $value `->` type($result) attr-dict
        }];
}

class Pylir_BinOp<string name, list<OpTrait> traits = []> : Pylir_Op<name,traits> {
    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
          $lhs `:` type($lhs) `,` $rhs `:` type($rhs) `->` type($result) attr-dict
        }];
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

def Pylir_ICmpOp : Pylir_Op<"icmp",[SameTypeOperands, NoSideEffect]> {
    let arguments = (ins Pylir_CmpPredicateAttr:$predicate, Pylir_Integer:$lhs, Pylir_Integer:$rhs);
    let results = (outs Pylir_Bool:$result);

    let assemblyFormat = "$predicate $lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict";

    let hasFolder = 1;
}

class Pylir_FBinOp<string name, list<OpTrait> traits = []> :
    Pylir_BinOp<name,!listconcat(traits,[SameOperandsAndResultType, NoSideEffect])> {
    let arguments = (ins Pylir_Float:$lhs, Pylir_Float:$rhs);
    let results = (outs Pylir_Float:$result);
}

def Pylir_FAddOp : Pylir_FBinOp<"fadd", [Commutative]>;

def Pylir_FSubOp : Pylir_FBinOp<"fsub">;

def Pylir_FMulOp : Pylir_FBinOp<"fmul", [Commutative]>;

def Pylir_FDivOp : Pylir_FBinOp<"ftruediv">;

def Pylir_FFloorDivOp : Pylir_FBinOp<"ffloordiv">;

def Pylir_FModOp : Pylir_FBinOp<"fmod">;

def Pylir_FCmpOp : Pylir_Op<"fcmp",[SameTypeOperands, NoSideEffect]> {
    let arguments = (ins Pylir_CmpPredicateAttr:$predicate, Pylir_Float:$lhs, Pylir_Float:$rhs);
    let results = (outs Pylir_Bool:$result);

    let assemblyFormat = "$predicate $lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict";

    let hasFolder = 1;
}

class Pylir_BBinOp<string name, list<OpTrait> traits = []> :
    Pylir_BinOp<name,!listconcat(traits,[SameOperandsAndResultType, NoSideEffect])> {

    let arguments = (ins Pylir_Bool:$lhs, Pylir_Bool:$rhs);
    let results = (outs Pylir_Bool:$result);
}

def Pylir_BAndOp : Pylir_BBinOp<"band", [Commutative]>;

def Pylir_BXorOp : Pylir_BBinOp<"bxor", [Commutative]>;

def Pylir_BOrOp : Pylir_BBinOp<"bor", [Commutative]>;

def Pylir_BNegOp : Pylir_Op<"bneg",[SameOperandsAndResultType, Involution]> {

    let arguments = (ins Pylir_Bool:$input);
    let results = (outs Pylir_Bool:$result);

    let assemblyFormat = [{
        $input `:` type($input) `->` type($result) attr-dict
    }];
}

class Pylir_Cast<string name, list<OpTrait> traits = []>
    : Pylir_Op<name,!listconcat(traits,[NoSideEffect, DeclareOpInterfaceMethods<CastOpInterface>])> {
    let assemblyFormat = [{
          $input `:` type($input) `->` type($result) attr-dict
        }];
}

def Pylir_ItoFOp : Pylir_Cast<"itof"> {
    let arguments = (ins Pylir_Integer:$input);
    let results = (outs Pylir_Float:$result);
}

def Pylir_BtoIOp : Pylir_Cast<"btoi"> {
    let arguments = (ins Pylir_Bool:$input);
    let results = (outs Pylir_Float:$result);
}

def Pylir_BtoI1Op : Pylir_Cast<"btoI1"> {
    let arguments = (ins Pylir_Bool:$input);
    let results = (outs I1:$result);

    let hasFolder = 0;
}

def Pylir_ToVariantOp : Pylir_Cast<"toVariant"> {
    let arguments = (ins AnyPythonType:$input);
    let results = (outs Pylir_Variant:$result);
}

def Pylir_MakeListOp : Pylir_Op<"makeList", [NoSideEffect]> {
    let arguments = (ins Variadic<AnyPythonType>:$input);
    let results = (outs Pylir_List:$result);

    let assemblyFormat = [{
              $input `:` type($input) `->` type($result) attr-dict
            }];
}

def Pylir_MakeTupleOp : Pylir_Op<"makeTuple", [NoSideEffect]> {
    let arguments = (ins Variadic<AnyPythonType>:$input);
    let results = (outs Pylir_Tuple:$result);

    let assemblyFormat = [{
              $input `:` type($input) `->` type($result) attr-dict
            }];
}

def Pylir_TupleToListOp : Pylir_Cast<"tupleToList"> {
    let arguments = (ins Pylir_Tuple:$input);
    let results = (outs Pylir_List:$result);
}

def Pylir_ListToTupleOp : Pylir_Cast<"listToTuple"> {
    let arguments = (ins Pylir_List:$input);
    let results = (outs Pylir_Tuple:$result);
}

def Pylir_TypeOfOp : Pylir_Op<"typeOf", [NoSideEffect]> {
    let arguments = (ins AnyPythonType:$input);
    let results = (outs AnyPythonType:$result);

    let assemblyFormat = [{
                  $input `:` type($input) `->` type($result) attr-dict
                }];

    let hasFolder = 0;
}

def Pylir_GetAttrOp : Pylir_Op<"getAttr", [NoSideEffect]> {
    let arguments = (ins AnyPythonType:$input, StrAttr:$member);
    let results = (outs AnyPythonType:$result);

    let assemblyFormat = [{
                      $member `from` $input `:` type($input) `->` type($result) attr-dict
                    }];

    let hasFolder = 0;

    let builders = [
            OpBuilderDAG<(ins "mlir::Value":$value, "mlir::StringAttr":$member),
            [{ build($_builder, $_state, ::pylir::Dialect::UnknownType::get($_state.getContext()), value, member); }]>];
}

def Pylir_IdOp : Pylir_Op<"id", [NoSideEffect]> {
    let arguments = (ins AnyPythonType:$input);
    let results = (outs Pylir_Integer:$result);

    let assemblyFormat = [{
                      $input `:` type($input) attr-dict
                    }];

    let hasFolder = 0;
    let hasFolder = 0;
}

def Pylir_CallOp : Pylir_Op<"call", [DeclareOpInterfaceMethods<CallOpInterface>]> {
    let arguments = (ins AnyPythonType:$callee, Variadic<AnyPythonType>:$operands);
    let results = (outs AnyPythonType:$result);

    let assemblyFormat = "$callee `:` type($callee)  `(` $operands `)` `->` functional-type($operands, results) attr-dict";

    let hasFolder = 0;
}

def Pylir_RaiseOp : Pylir_Op<"raise", [Terminator]> {
    let hasFolder = 0;

    let arguments = (ins AnyPythonType:$exception);
    let results = (outs);

    let assemblyFormat = "$exception `:` type($exception) attr-dict";
}

def Pylir_GetItemOp : Pylir_Op<"getItem>",[NoSideEffect]> {
    let arguments = (ins AnySequenceType:$sequence,Pylir_Integer:$index);
    let results = (outs AnyPythonType:$result);

    let assemblyFormat = [{
                          $sequence `:` type($sequence) `[` $index `]` `->` type($result) attr-dict
                        }];
}

#endif
