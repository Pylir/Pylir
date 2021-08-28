
#ifndef PYLIR_PY_OPS_TABLEGEN
#define PYLIR_PY_OPS_TABLEGEN

include "pylir/Optimizer/PylirPy/IR/PylirPyBase.td"
include "pylir/Optimizer/PylirPy/IR/PylirPyTypes.td"
include "pylir/Optimizer/PylirPy/IR/PylirPyAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def DynamicType : TypeAlias<PylirPy_DynamicType>;

class PylirPy_Op<string mneomic, list<OpTrait> traits = []> :
    Op<PylirPy_Dialect, mneomic, traits> {
    let results = (outs DynamicType:$result);
}

def PylirPy_Constant : PylirPy_Op<"constant", [ConstantLike, NoSideEffect]> {
    let arguments = (ins AnyPyAttribute:$constant);

    let assemblyFormat = [{
        $constant attr-dict
    }];

    let hasFolder = 1;
}

def PylirPy_GetAttr : PylirPy_Op<"getAttr"> {
    let arguments = (ins DynamicType:$object, StrAttr:$attribute);

    let assemblyFormat = [{
        $attribute `from` $object attr-dict
    }];

    let hasFolder = 1;
}

def PylirPy_GetItem : PylirPy_Op<"getItem"> {
    let arguments = (ins DynamicType:$object, DynamicType:$index);

    let assemblyFormat = [{
        $object `[` $index `]` attr-dict
    }];

    let hasFolder = 1;
}

#endif
