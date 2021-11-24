
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

def DynamicType : Type<CPred<"$_self.isa<::pylir::Py::DynamicType>()">, "python dynamic type", "::pylir::Py::DynamicType"> {
    let builderCall = "$_builder.getType<::pylir::Py::DynamicType>()";
}

def MemoryType : TypeAlias<PylirMem_MemoryType>;

def ObjectAttr : Attr<CPred<"$_self.isa<::pylir::Py::ObjectAttr>()">, "python object attribute"> {
    let storageType = "::pylir::Py::ObjectAttr";
    let returnType = "::pylir::Py::ObjectAttr";
    let convertFromStorage = "$_self";
}

class PylirMem_Op<string mnemonic, list<OpTrait> traits = []> :
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

def PylirMem_InitIntOp : PylirMem_Op<"initInt"> {
    let results = (outs DynamicType:$result);
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, AnySignlessInteger:$initializer);

    let assemblyFormat = "$memory `=` $initializer `:` type($initializer) attr-dict";
}

def PylirMem_InitFloatOp : PylirMem_Op<"initFloat"> {
    let results = (outs DynamicType:$result);
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, F64:$initializer);

    let assemblyFormat = "$memory `=` $initializer attr-dict";
}

def PylirMem_InitTupleOp : PylirMem_Op<"initTuple", [DeclareOpInterfaceMethods<InferTypeOpInterface>, AttrSizedOperandSegments]> {
    let results = (outs DynamicType:$result);
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, Variadic<DynamicType>:$initializer, Arg<Optional<DynamicType>, "", [MemWrite]>:$reuse);

    let assemblyFormat = "$memory `=` `(` $initializer `)` (`reuse` $reuse^)? attr-dict";
}

def PylirMem_InitListOp : PylirMem_Op<"initList", [DeclareOpInterfaceMethods<InferTypeOpInterface>, AttrSizedOperandSegments]> {
    let results = (outs DynamicType:$result);
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, Variadic<DynamicType>:$initializer, Arg<Optional<DynamicType>, "", [MemWrite]>:$reuse);

    let assemblyFormat = "$memory `=` `[` $initializer `]` (`reuse` $reuse^)? attr-dict";
}

def PylirMem_InitSetOp : PylirMem_Op<"initSet", [DeclareOpInterfaceMethods<InferTypeOpInterface>, AttrSizedOperandSegments]> {
    let results = (outs DynamicType:$result);
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, Variadic<DynamicType>:$initializer, Arg<Optional<DynamicType>, "", [MemWrite]>:$reuse);

    let assemblyFormat = "$memory `=` `{` $initializer `}` (`reuse` $reuse^)? attr-dict";
}

def PylirMem_InitDictOp : PylirMem_Op<"initDict"> {
    let results = (outs DynamicType:$result);
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory);

    let assemblyFormat = "$memory attr-dict";
}

def PylirMem_InitFunctionOp : PylirMem_Op<"initFunction"> {
    let results = (outs DynamicType:$result);
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory, SymbolRefAttr:$initializer);

    let assemblyFormat = "$memory `=` $initializer attr-dict";
}

def PylirMem_InitObjectOp : PylirMem_Op<"initObject"> {
    let results = (outs DynamicType:$result);
    let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory);

    let assemblyFormat = "$memory attr-dict";
}

#endif
