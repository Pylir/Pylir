
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
                         DefaultValuedAttr<I32ArrayAttr, "{}">:$iterExpansion
                         );

    let assemblyFormat = [{
        custom<IterArguments>($arguments, $iterExpansion) attr-dict
    }];

    let hasFolder = 1;
}

#endif
