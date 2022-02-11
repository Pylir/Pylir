
#ifndef PYLIR_OPS_TABLEGEN
#define PYLIR_OPS_TABLEGEN

include "pylir/Optimizer/PylirMem/IR/PylirMemTypes.td"
include "pylir/Optimizer/PylirMem/IR/PylirMemAttributes.td"
include "pylir/Optimizer/PylirPy/IR/PylirPyTraits.td"
include "pylir/Optimizer/PylirPy/IR/PylirPyTypes.td"
include "pylir/Optimizer/PylirPy/IR/PylirPyAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"

defvar ObjectAttr = PylirPy_ObjectAttr;
defvar MemoryType = PylirMem_MemoryType;

class PylirMem_Op<string mnemonic, list<Trait> traits = []> :
    Op<PylirMem_Dialect, mnemonic, traits>;

def PylirMem_GCAllocObjectOp : PylirMem_Op<"gcAllocObject"> {
    let results = (outs Arg<MemoryType, "", [MemAlloc]>:$result);
    let arguments = (ins DynamicType:$typeObj);

    let assemblyFormat = "$typeObj attr-dict";
}

def PylirMem_StackAllocObjectOp : PylirMem_Op<"stackAllocObject"> {
    let results = (outs Arg<MemoryType, "", [MemAlloc<AutomaticAllocationScopeResource>]>:$result);
    let arguments = (ins ObjectAttr:$typeObj);

    let assemblyFormat = "$typeObj attr-dict";
}

def PylirMem_GCAllocTupleOp : PylirMem_Op<"gcAllocTuple"> {
    let results = (outs Arg<MemoryType, "", [MemAlloc]>:$result);
    let arguments = (ins DynamicType:$typeObj, Index:$length);

    let assemblyFormat = "$typeObj `[` $length `]` attr-dict";
}

def PylirMem_StackAllocTupleOp : PylirMem_Op<"stackAllocTuple"> {
    let results = (outs Arg<MemoryType, "", [MemAlloc<AutomaticAllocationScopeResource>]>:$result);
    let arguments = (ins DynamicType:$typeObj, IndexAttr:$length);

    let assemblyFormat = "$typeObj `[` $length `]` attr-dict";
}

def PylirMem_InitIntOp : PylirMem_Op<"initInt", [AlwaysBound]> {
    let results = (outs DynamicType:$result);
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, IntegerLike:$initializer);

    let assemblyFormat = "$memory `to` $initializer `:` type($initializer) attr-dict";
}

def PylirMem_InitIntAddOp : PylirMem_Op<"initIntAdd", [AlwaysBound]> {
    let results = (outs DynamicType:$result);
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, DynamicType:$lhs, DynamicType:$rhs);

    let assemblyFormat = "$memory `to` $lhs `+` $rhs attr-dict";
}

def PylirMem_InitFloatOp : PylirMem_Op<"initFloat", [AlwaysBound]> {
    let results = (outs DynamicType:$result);
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, F64:$initializer);

    let assemblyFormat = "$memory `to` $initializer attr-dict";
}

def PylirMem_InitTupleOp : PylirMem_Op<"initTuple", [DeclareOpInterfaceMethods<InferTypeOpInterface>, AlwaysBound]> {
    let results = (outs DynamicType:$result);
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, Variadic<DynamicType>:$initializer);

    let assemblyFormat = "$memory `to` ` ` `(` $initializer `)` attr-dict";
}

def PylirMem_InitTuplePrependOp : PylirMem_Op<"initTuplePrepend", [AlwaysBound]> {
    let results = (outs DynamicType:$result);
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, DynamicType:$element, DynamicType:$tuple);

    let assemblyFormat = "$memory `to` ` ` `(` $element `,` $tuple `)` attr-dict";
}

def PylirMem_InitTuplePopFrontOp : PylirMem_Op<"initTuplePopFront", [AlwaysBound]> {
    let results = (outs DynamicType:$result);
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, DynamicType:$tuple);

    let assemblyFormat = "$memory `to` ` ` `(` $tuple `)` attr-dict";
}

def PylirMem_InitTupleFromListOp : PylirMem_Op<"initTupleFromList"> {
    let results = (outs DynamicType:$result);
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, DynamicType:$initializer);

    let assemblyFormat = "$memory `to` ` ` `(` `*` $initializer `)` attr-dict";
}

def PylirMem_InitListOp : PylirMem_Op<"initList", [DeclareOpInterfaceMethods<InferTypeOpInterface>, AlwaysBound]> {
    let results = (outs DynamicType:$result);
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, Variadic<DynamicType>:$initializer);

    let assemblyFormat = "$memory `to` ` ` `[` $initializer `]` attr-dict";
}

def PylirMem_InitSetOp : PylirMem_Op<"initSet", [DeclareOpInterfaceMethods<InferTypeOpInterface>, AlwaysBound]> {
    let results = (outs DynamicType:$result);
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, Variadic<DynamicType>:$initializer);

    let assemblyFormat = "$memory `to` ` ` `{` $initializer `}` attr-dict";
}

def PylirMem_InitDictOp : PylirMem_Op<"initDict", [AlwaysBound]> {
    let results = (outs DynamicType:$result);
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory);

    let assemblyFormat = "$memory attr-dict";
}

def PylirMem_InitStrOp : PylirMem_Op<"initStr", [DeclareOpInterfaceMethods<InferTypeOpInterface>, AlwaysBound]> {
    let results = (outs DynamicType:$result);
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, Variadic<DynamicType>:$strings);

    let assemblyFormat = "$memory (`to` $strings^)? attr-dict";
}

def PylirMem_InitStrFromIntOp : PylirMem_Op<"initStrFromInt", [AlwaysBound]> {
    let results = (outs DynamicType:$result);
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, DynamicType:$integer);

    let assemblyFormat = "$memory `to` $integer attr-dict";
}

def PylirMem_InitFuncOp : PylirMem_Op<"initFunc", [AlwaysBound]> {
    let results = (outs DynamicType:$result);
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, FlatSymbolRefAttr:$initializer);

    let assemblyFormat = "$memory `to` $initializer attr-dict";
}

def PylirMem_InitObjectOp : PylirMem_Op<"initObject", [AlwaysBound]> {
    let results = (outs DynamicType:$result);
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory);

    let assemblyFormat = "$memory attr-dict";
}

#endif
