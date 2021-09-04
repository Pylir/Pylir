
#ifndef PYLIR_PY_OPS_TABLEGEN
#define PYLIR_PY_OPS_TABLEGEN

include "pylir/Optimizer/PylirPy/IR/PylirPyBase.td"
include "pylir/Optimizer/PylirPy/IR/PylirPyTypes.td"
include "pylir/Optimizer/PylirPy/IR/PylirPyAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/CallInterfaces.td"

def DynamicType : TypeAlias<PylirPy_DynamicType>;

class PylirPy_Op<string mneomic, list<OpTrait> traits = []> :
    Op<PylirPy_Dialect, mneomic, traits> {
    let results = (outs DynamicType:$result);
}

def PylirPy_ConstantOp : PylirPy_Op<"constant", [ConstantLike, NoSideEffect]> {
    let arguments = (ins AnyPyAttribute:$constant);

    let assemblyFormat = [{
        $constant attr-dict
    }];

    let hasFolder = 1;
}

def PylirPy_GetAttrOp : PylirPy_Op<"getAttr"> {
    let arguments = (ins DynamicType:$object, StrAttr:$attribute);

    let assemblyFormat = [{
        $attribute `from` $object attr-dict
    }];

    let hasFolder = 1;
}

def PylirPy_GetItemOp : PylirPy_Op<"getItem"> {
    let arguments = (ins DynamicType:$object, DynamicType:$index);

    let assemblyFormat = [{
        $object `[` $index `]` attr-dict
    }];

    let hasFolder = 1;
}

def PylirPy_CallOp : PylirPy_Op<"call", [DeclareOpInterfaceMethods<CallOpInterface>]> {
    let arguments = (ins DynamicType:$callee,
                         Variadic<DynamicType>:$arguments,
                         I32ArrayAttr:$iterExpansion,
                         I32ArrayAttr:$mappingExpansion
                         );

    let assemblyFormat = [{
        $callee custom<ExpandArguments>($arguments, $iterExpansion, $mappingExpansion) attr-dict
    }];
}

def PylirPy_MakeTupleOp : PylirPy_Op<"makeTuple"> {
    let arguments = (ins Variadic<DynamicType>:$arguments,
                         I32ArrayAttr:$iterExpansion
                         );

    let assemblyFormat = [{
        custom<IterArguments>($arguments, $iterExpansion) attr-dict
    }];

    let hasFolder = 1;
}

def PylirPy_PowerOp : PylirPy_Op<"pow"> {
    let arguments = (ins DynamicType:$base, DynamicType:$exponent, Optional<DynamicType>:$mod);

    let assemblyFormat = [{
        $base `to` $exponent (`mod` $mod^)? attr-dict
    }];

    let hasFolder = 1;
}

// Unary ops

class PylirPy_UnaryOp<string name, list<OpTrait> traits = []> : PylirPy_Op<name, traits> {
    let arguments = (ins DynamicType:$input);

    let assemblyFormat = "$input attr-dict";

    let hasFolder = 1;
}

def PylirPy_NegOp : PylirPy_Op<"neg">;
def PylirPy_PosOp : PylirPy_Op<"pos">;
def PylirPy_InvertOp : PylirPy_Op<"invert">;

// Binary ops

class PylirPy_BinOp<string name, list<OpTrait> traits = []> : PylirPy_Op<name,traits> {
    let arguments = (ins DynamicType:$lhs, DynamicType:$rhs);

    let assemblyFormat = "$lhs `,` $rhs attr-dict";

    let hasFolder = 1;
}

def PylirPy_MulOp : PylirPy_BinOp<"mul">;
def PylirPy_MatMulOp : PylirPy_BinOp<"matMul"> {
    let hasFolder = 0;
}
def PylirPy_FloorDivOp : PylirPy_BinOp<"floorDiv">;
def PylirPy_TrueDivOp : PylirPy_BinOp<"trueDiv">;
def PylirPy_ModuloOp : PylirPy_BinOp<"mod">;
def PylirPy_AddOp : PylirPy_BinOp<"add">;
def PylirPy_SubOp : PylirPy_BinOp<"sub">;
def PylirPy_LShiftOp : PylirPy_BinOp<"lshift">;
def PylirPy_RShiftOp : PylirPy_BinOp<"rshift">;
def PylirPy_AndOp : PylirPy_BinOp<"and">;
def PylirPy_OrOp : PylirPy_BinOp<"or">;
def PylirPy_XorOp : PylirPy_BinOp<"xor">;
def PylirPy_LessOp : PylirPy_BinOp<"less"> {
    let hasFolder = 0;
}
def PylirPy_LessEqualOp : PylirPy_BinOp<"lessEqual"> {
    let hasFolder = 0;
}
def PylirPy_GreaterOp : PylirPy_BinOp<"greater"> {
    let hasFolder = 0;
}
def PylirPy_GreaterEqualOp : PylirPy_BinOp<"greaterEqual"> {
    let hasFolder = 0;
}
def PylirPy_EqualOp : PylirPy_BinOp<"equal"> {
    let hasFolder = 0;
}
def PylirPy_NotEqualOp : PylirPy_BinOp<"notEqual"> {
    let hasFolder = 0;
}
def PylirPy_IsOp : PylirPy_BinOp<"is"> {
    let hasFolder = 0;
}
def PylirPy_InOp : PylirPy_BinOp<"in"> {
    let hasFolder = 0;
}

#endif
