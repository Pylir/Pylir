
#ifndef PYLIR_OPS_TABLEGEN
#define PYLIR_OPS_TABLEGEN

include "PylirTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Pylir_Op<string mneomic, list<OpTrait> traits = []> :
    Op<Pylir_Dialect, mneomic, traits> {
    let hasFolder = 1;
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
    : Pylir_BinOp<name,!listconcat(traits,[NoSideEffect])> {
    let arguments = (ins Pylir_Integer:$lhs, Pylir_Integer:$rhs);
    let results = (outs Pylir_Integer:$result);
}

def Pylir_IAddOp : Pylir_IBinOp<"iadd", [SameOperandsAndResultType, Commutative]>;

def Pylir_ISubOp : Pylir_IBinOp<"isub", [SameOperandsAndResultType]>;

def Pylir_IMulOp : Pylir_IBinOp<"imul", [SameOperandsAndResultType, Commutative]>;

def Pylir_IDivOp : Pylir_IBinOp<"itruediv", [SameTypeOperands]> {
    let results = (outs Pylir_Float:$result);

    let builders = [
                OpBuilderDAG<(ins "mlir::Value":$lhs,"mlir::Value":$rhs),
                [{ build($_builder, $_state, FloatType::get($_state.getContext()), lhs, rhs); }]>];
}

def Pylir_IFloorDivOp : Pylir_BinOp<"ifloordiv", [SameOperandsAndResultType]>;

def Pylir_IModOp : Pylir_BinOp<"imod", [SameOperandsAndResultType]>;

def Pylir_INegOp : Pylir_Op<"ineg",[SameOperandsAndResultType, Involution]> {

    let arguments = (ins Pylir_Integer:$input);
    let results = (outs Pylir_Integer:$result);

    let assemblyFormat = [{
        $input `:` type($input) `->` type($result) attr-dict
    }];
}

def Pylir_IShlOp : Pylir_BinOp<"ishl",[SameOperandsAndResultType]>;

def Pylir_IShrOp : Pylir_BinOp<"ishr",[SameOperandsAndResultType]>;

def Pylir_IAndOp : Pylir_BinOp<"iand", [SameOperandsAndResultType, Commutative]>;

def Pylir_IXorOp : Pylir_BinOp<"ixor", [SameOperandsAndResultType, Commutative]>;

def Pylir_IOrOp : Pylir_BinOp<"ior", [SameOperandsAndResultType, Commutative]>;

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

class Pylir_Cast<string name, list<OpTrait> traits = []> : Pylir_Op<name,traits> {
    let assemblyFormat = [{
          $input `:` type($input) }] # "`" # name # "`" # [{ type($result) attr-dict
        }];
}

def Pylir_ItoF : Pylir_Cast<"itof",[NoSideEffect]> {
    let arguments = (ins Pylir_Integer:$input);
    let results = (outs Pylir_Float:$result);

    let builders = [
            OpBuilderDAG<(ins "mlir::Value":$value),
            [{ build($_builder, $_state, FloatType::get($_state.getContext()), value); }]>];
}

#endif
