
#ifndef PYLIR_OPS_TABLEGEN
#define PYLIR_OPS_TABLEGEN

include "PylirTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Pylir_Op<string mneomic, list<OpTrait> traits = []> :
    Op<Pylir_Dialect, mneomic, traits>;

class Pylir_BinOp<string name> : Pylir_Op<name> {
    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
          $lhs `:` type($lhs) `,` $rhs `:` type($rhs) `->` type($result) attr-dict
        }];
}

def Pylir_AddOp : Pylir_BinOp<"add">;

def Pylir_SubOp : Pylir_BinOp<"sub">;

def Pylir_MulOp : Pylir_BinOp<"mul"> {
    let hasFolder = 1;
}

def Pylir_MatMulOp : Pylir_BinOp<"matmul">;

def Pylir_DivOp : Pylir_BinOp<"truediv">;

def Pylir_IntDivOp : Pylir_BinOp<"floordiv">;

def Pylir_ModOp : Pylir_BinOp<"mod">;

def Pylir_DivModOp : Pylir_BinOp<"divmod">;

def Pylir_PowerOp : Pylir_Op<"power"> {
    let arguments = (ins AnyType:$lhs, AnyType:$rhs, Optional<AnyType>:$mod);
    let results = (outs AnyType:$res);

    let assemblyFormat = [{
        $lhs `:` type($lhs) `,` $rhs `:` type($rhs) (`mod` $mod^ `:` type($mod))? `->` type($res) attr-dict
    }];
}

def Pylir_LShiftOp : Pylir_BinOp<"lshift">;

def Pylir_RShiftOp : Pylir_BinOp<"rshift">;

def Pylir_AndOp : Pylir_BinOp<"and">;

def Pylir_XorOp : Pylir_BinOp<"xor">;

def Pylir_OrOp : Pylir_BinOp<"or">;

def Pylir_ConstantOp : Pylir_Op<"constant",[ConstantLike, NoSideEffect]> {
    let arguments = (ins AnyAttr:$value);
    let results = (outs AnyType:$result);

    let builders = [
        OpBuilderDAG<(ins "mlir::Attribute":$value),
        [{ build($_builder, $_state, value.getType(), value); }]>];

    let assemblyFormat = [{
            $value `->` type($result) attr-dict
        }];

    let hasFolder = 1;
}

#endif
