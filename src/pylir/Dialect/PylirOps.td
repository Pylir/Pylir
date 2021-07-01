
#ifndef PYLIR_OPS_TABLEGEN
#define PYLIR_OPS_TABLEGEN

include "PylirTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Pylir_Op<string mneomic, list<OpTrait> traits = []> :
    Op<Pylir_Dialect, mneomic, traits>;

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

class Pylir_BinOp<string name> : Pylir_Op<name> {
    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
          $lhs `:` type($lhs) `,` $rhs `:` type($rhs) `->` type($result) attr-dict
        }];
}

class Pylir_IBinOp<string name> : Pylir_BinOp<name> {
    let arguments = (ins Pylir_Integer:$lhs, Pylir_Integer:$rhs);
    let results = (outs Pylir_Integer:$result);
}

def Pylir_IAddOp : Pylir_IBinOp<"iadd">;

def Pylir_ISubOp : Pylir_IBinOp<"isub">;

def Pylir_IMulOp : Pylir_IBinOp<"imul">;

def Pylir_IDivOp : Pylir_IBinOp<"itruediv"> {
    let results = (outs Pylir_Float:$result);
}

def Pylir_IFloorDivOp : Pylir_BinOp<"ifloordiv">;

def Pylir_IModOp : Pylir_BinOp<"imod">;

class Pylir_FBinOp<string name> : Pylir_BinOp<name> {
    let arguments = (ins Pylir_Float:$lhs, Pylir_Float:$rhs);
    let results = (outs Pylir_Float:$result);
}

def Pylir_FAddOp : Pylir_IBinOp<"fadd">;

def Pylir_FSubOp : Pylir_IBinOp<"fsub">;

def Pylir_FMulOp : Pylir_IBinOp<"fmul">;

def Pylir_FDivOp : Pylir_IBinOp<"ftruediv"> {
    let results = (outs Pylir_Float:$result);
}

def Pylir_FFloorDivOp : Pylir_BinOp<"ffloordiv">;

def Pylir_FModOp : Pylir_BinOp<"fmod">;

class Pylir_Cast<string name> : Pylir_Op<name> {
    let assemblyFormat = [{
          $input `:` type($input) }] # "`" # name # "`" # [{ type($result) attr-dict
        }];
}

def Pylir_ItoF : Pylir_Cast<"itof"> {
    let arguments = (ins Pylir_Integer:$input);
    let results = (outs Pylir_Float:$result);
}

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

#endif
