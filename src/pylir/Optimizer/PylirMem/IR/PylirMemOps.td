// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
include "mlir/IR/EnumAttr.td"

defvar ObjectAttr = PylirPy_ObjectAttr;
defvar MemoryType = PylirMem_MemoryType;

class PylirMem_Op<string mnemonic, list<Trait> traits = []> :
    Op<PylirMem_Dialect, mnemonic, traits>;

def PylirMem_GCAllocObjectOp : PylirMem_Op<"gcAllocObject"> {
    let arguments = (ins DynamicType:$type_object);
    let results = (outs Arg<MemoryType, "", [MemAlloc]>:$result);

    let assemblyFormat = "$type_object attr-dict";
}

def PylirMem_StackAllocObjectOp : PylirMem_Op<"stackAllocObject"> {
    let arguments = (ins ObjectAttr:$type_object);
    let results = (outs Arg<MemoryType, "", [MemAlloc<AutomaticAllocationScopeResource>]>:$result);

    let assemblyFormat = "$type_object attr-dict";
}

def PylirMem_GCAllocTupleOp : PylirMem_Op<"gcAllocTuple"> {
    let arguments = (ins DynamicType:$type_object, Index:$length);
    let results = (outs Arg<MemoryType, "", [MemAlloc]>:$result);

    let assemblyFormat = "$type_object `[` $length `]` attr-dict";
}

def PylirMem_StackAllocTupleOp : PylirMem_Op<"stackAllocTuple"> {
    let arguments = (ins DynamicType:$type_object, IndexAttr:$length);
    let results = (outs Arg<MemoryType, "", [MemAlloc<AutomaticAllocationScopeResource>]>:$result);

    let assemblyFormat = "$type_object `[` $length `]` attr-dict";
}

def PylirMem_InitIntSignedOp : PylirMem_Op<"initIntSigned", [AlwaysBound]> {
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, Index:$initializer);
    let results = (outs DynamicType:$result);

    let assemblyFormat = "$memory `to` $initializer attr-dict";
}

def PylirMem_InitIntUnsignedOp : PylirMem_Op<"initIntUnsigned", [AlwaysBound]> {
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, Index:$initializer);
    let results = (outs DynamicType:$result);

    let assemblyFormat = "$memory `to` $initializer attr-dict";
}

def PylirMem_InitIntAddOp : PylirMem_Op<"initIntAdd", [AlwaysBound]> {
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, DynamicType:$lhs, DynamicType:$rhs);
    let results = (outs DynamicType:$result);

    let assemblyFormat = "$memory `to` $lhs `+` $rhs attr-dict";
}

def PylirMem_InitFloatOp : PylirMem_Op<"initFloat", [AlwaysBound]> {
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, F64:$initializer);
    let results = (outs DynamicType:$result);

    let assemblyFormat = "$memory `to` $initializer attr-dict";
}

def PylirMem_InitTupleOp : PylirMem_Op<"initTuple", [AlwaysBound]> {
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, Variadic<DynamicType>:$initializer);
    let results = (outs DynamicType:$result);

    let assemblyFormat = "$memory `to` ` ` `(` $initializer `)` attr-dict";
}

def PylirMem_InitTuplePrependOp : PylirMem_Op<"initTuplePrepend", [AlwaysBound]> {
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, DynamicType:$element, DynamicType:$tuple);
    let results = (outs DynamicType:$result);

    let assemblyFormat = "$memory `to` ` ` `(` $element `,` $tuple `)` attr-dict";
}

def PylirMem_InitTupleDropFrontOp : PylirMem_Op<"initTupleDropFront", [AlwaysBound]> {
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, Index:$count, DynamicType:$tuple);
    let results = (outs DynamicType:$result);

    let assemblyFormat = "$memory `to` ` ` $count `(` $tuple `)` attr-dict";
}

def PylirMem_InitTupleCopyOp : PylirMem_Op<"initTupleCopy"> {
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, DynamicType:$initializer);
    let results = (outs DynamicType:$result);

    let assemblyFormat = "$memory `to` ` ` `(` `*` $initializer `)` attr-dict";
}

def PylirMem_InitTupleFromListOp : PylirMem_Op<"initTupleFromList"> {
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, DynamicType:$initializer);
    let results = (outs DynamicType:$result);

    let assemblyFormat = "$memory `to` ` ` `(` `*` $initializer `)` attr-dict";
}

def PylirMem_InitListOp : PylirMem_Op<"initList", [AlwaysBound]> {
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, Variadic<DynamicType>:$initializer);
    let results = (outs DynamicType:$result);

    let assemblyFormat = "$memory `to` ` ` `[` $initializer `]` attr-dict";
}

def PylirMem_InitSetOp : PylirMem_Op<"initSet", [AlwaysBound]> {
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, Variadic<DynamicType>:$initializer);
    let results = (outs DynamicType:$result);

    let assemblyFormat = "$memory `to` ` ` `{` $initializer `}` attr-dict";
}

def PylirMem_InitDictOp : PylirMem_Op<"initDict", [AlwaysBound]> {
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory);
    let results = (outs DynamicType:$result);

    let assemblyFormat = "$memory attr-dict";
}

def PylirMem_InitStrOp : PylirMem_Op<"initStr", [AlwaysBound]> {
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, Variadic<DynamicType>:$strings);
    let results = (outs DynamicType:$result);

    let assemblyFormat = "$memory (`to` $strings^)? attr-dict";
}

def PylirMem_InitStrFromIntOp : PylirMem_Op<"initStrFromInt", [AlwaysBound]> {
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, DynamicType:$integer);
    let results = (outs DynamicType:$result);

    let assemblyFormat = "$memory `to` $integer attr-dict";
}

def PylirMem_InitFuncOp : PylirMem_Op<"initFunc", [AlwaysBound]> {
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, FlatSymbolRefAttr:$initializer);
    let results = (outs DynamicType:$result);

    let assemblyFormat = "$memory `to` $initializer attr-dict";
}

def PylirMem_InitObjectOp : PylirMem_Op<"initObject", [AlwaysBound]> {
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory);
    let results = (outs DynamicType:$result);

    let assemblyFormat = "$memory attr-dict";
}

#endif
