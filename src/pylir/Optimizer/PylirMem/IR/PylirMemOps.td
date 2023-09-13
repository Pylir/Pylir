// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_OPS_TABLEGEN
#define PYLIR_OPS_TABLEGEN

include "pylir/Optimizer/Interfaces/CaptureInterface.td"
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

def PylirMem_GCAllocObjectOp : PylirMem_Op<"gcAllocObject",
  [DeclareOpInterfaceMethods<CaptureInterface>]> {

  let arguments = (ins
    Arg<DynamicType, "", [Captured]>:$type_object,
    Index:$trailing_items
  );
  let results = (outs Arg<MemoryType, "", [MemAlloc]>:$result);

  let description = [{
    Allocates an object with the layout type and type given by the
    `$type_object` with `$trailing_items`.
    These trailing items correspond to the amount of elements in a tuple for
    tuple objects, for all other objects they correspond to the amount of slots
    allocated.

    Items allocated by the operation are tracked by the garbage collector and
    guaranteed to be life as long as they are reachable.
  }];

  let assemblyFormat = "$type_object`[` $trailing_items `]` attr-dict";
}

/// Object layouts known to the compiler with a static size. May be replaced
/// with some kind of type descriptors if we ever introduce custom object
/// layouts as an optimization.
def PylirMem_LayoutTypeAttr : I32EnumAttr<"LayoutType",
  "Enum of types with known object layouts", [
  I32EnumAttrCase<"Object", 0, "object">,
  I32EnumAttrCase<"Type", 1, "type">,
  I32EnumAttrCase<"Float", 2, "float">,
  I32EnumAttrCase<"Function", 3, "function">,
  I32EnumAttrCase<"Tuple", 4, "tuple">,
  I32EnumAttrCase<"List", 5, "list">,
  I32EnumAttrCase<"String", 6, "string">,
  I32EnumAttrCase<"Dict", 7, "dict">,
  I32EnumAttrCase<"Int", 8, "int">,
  I32EnumAttrCase<"BaseException", 9>,
]> {
  let cppNamespace = "pylir::Mem";
}

def PylirMem_StackAllocObjectOp : PylirMem_Op<"stackAllocObject", [
  DeclareOpInterfaceMethods<CaptureInterface>]> {

  let arguments = (ins Arg<DynamicType, "", [Captured]>:$type_object,
             PylirMem_LayoutTypeAttr:$layout,
             IndexAttr:$trailing_items);

  let results = (outs
    Arg<MemoryType, "", [MemAlloc<AutomaticAllocationScopeResource>]>:$result);

  let description = [{
    Allocates an object with the layout given by `$layout` with
    `$trailing_items` and the type `$type_object`.
    These trailing items correspond to the amount of elements in a tuple for
    tuple objects, for all other objects they correspond to the amount of slots
    allocated.

    Due to the use of attributes for the layout and amount of trailing items,
    the amount of memory allocated by this operation is known ahead of time and
    statically allocated within a functions frame.
    The lifetime of this object is therefore also equal to the frame of the
    function call. If any references to a destroyed object still exist it is
    undefined behaviour.
  }];

  let assemblyFormat = "$layout $type_object`[`$trailing_items`]` attr-dict";
}

def PylirMem_InitIntSignedOp : PylirMem_Op<"initIntSigned",
  [AlwaysBound, NoCaptures]> {
  let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory,
    Index:$initializer);
  let results = (outs DynamicType:$result);

  let assemblyFormat = "$memory `to` $initializer attr-dict";
}

def PylirMem_InitIntUnsignedOp : PylirMem_Op<"initIntUnsigned",
  [AlwaysBound, NoCaptures]> {
  let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory,
    Index:$initializer);
  let results = (outs DynamicType:$result);

  let assemblyFormat = "$memory `to` $initializer attr-dict";
}

def PylirMem_InitIntAddOp : PylirMem_Op<"initIntAdd",
  [AlwaysBound, NoCaptures]> {
  let arguments = (ins
    Arg<MemoryType, "", [MemWrite]>:$memory,
    DynamicType:$lhs,
    DynamicType:$rhs);
  let results = (outs DynamicType:$result);

  let assemblyFormat = "$memory `to` $lhs `+` $rhs attr-dict";
}

def PylirMem_InitFloatOp : PylirMem_Op<"initFloat",
  [AlwaysBound, NoCaptures]> {
  let arguments = (ins
    Arg<MemoryType, "", [MemWrite]>:$memory,
    F64:$initializer);
  let results = (outs DynamicType:$result);

  let assemblyFormat = "$memory `to` $initializer attr-dict";
}

def PylirMem_InitTupleOp : PylirMem_Op<"initTuple", [AlwaysBound,
  DeclareOpInterfaceMethods<CaptureInterface>]> {

  let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory,
             Arg<Variadic<DynamicType>, "", [Captured]>:$initializer);

  let results = (outs DynamicType:$result);

  let assemblyFormat = [{
    $memory `to` ` ` `(` (`)`) : (`` $initializer^ `)`)?  attr-dict
  }];
}

def PylirMem_InitTuplePrependOp : PylirMem_Op<"initTuplePrepend", [AlwaysBound,
                                   DeclareOpInterfaceMethods<CaptureInterface>]> {
  let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory,
             Arg<DynamicType, "", [Captured]>:$element,
             DynamicType:$tuple);

  let results = (outs DynamicType:$result);

  let assemblyFormat = "$memory `to` ` ` `(` $element `,` $tuple `)` attr-dict";
}

def PylirMem_InitTupleDropFrontOp : PylirMem_Op<"initTupleDropFront",
  [AlwaysBound, NoCaptures]> {
  let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory,
    Index:$count, DynamicType:$tuple);
  let results = (outs DynamicType:$result);

  let assemblyFormat = "$memory `to` ` ` $count `(` $tuple `)` attr-dict";
}

def PylirMem_InitTupleCopyOp : PylirMem_Op<"initTupleCopy", [NoCaptures]> {
  let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory,
    DynamicType:$initializer);
  let results = (outs DynamicType:$result);

  let assemblyFormat = "$memory `to` ` ` `(` `*` $initializer `)` attr-dict";
}

def PylirMem_InitTupleFromListOp : PylirMem_Op<"initTupleFromList",
  [NoCaptures]> {
  let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory,
    DynamicType:$initializer);
  let results = (outs DynamicType:$result);

  let assemblyFormat = "$memory `to` ` ` `(` `*` $initializer `)` attr-dict";
}

def PylirMem_InitListOp : PylirMem_Op<"initList", [AlwaysBound,
  DeclareOpInterfaceMethods<CaptureInterface>]> {

  let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory,
             Arg<Variadic<DynamicType>, "", [Captured]>:$initializer);

  let results = (outs DynamicType:$result);

  let assemblyFormat = "$memory `to` ` ` `[` $initializer `]` attr-dict";
}

def PylirMem_InitSetOp : PylirMem_Op<"initSet", [AlwaysBound,
  DeclareOpInterfaceMethods<CaptureInterface>]> {

  let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory,
             Arg<Variadic<DynamicType>, "", [Captured]>:$initializer);

  let results = (outs DynamicType:$result);

  let assemblyFormat = "$memory `to` ` ` `{` $initializer `}` attr-dict";
}

def PylirMem_InitDictOp : PylirMem_Op<"initDict", [AlwaysBound, NoCaptures]> {
  let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory);
  let results = (outs DynamicType:$result);

  let assemblyFormat = "$memory attr-dict";
}

def PylirMem_InitStrOp : PylirMem_Op<"initStr", [AlwaysBound, NoCaptures]> {
  let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory,
    Variadic<DynamicType>:$strings);
  let results = (outs DynamicType:$result);

  let assemblyFormat = "$memory (`to` $strings^)? attr-dict";
}

def PylirMem_InitStrFromIntOp : PylirMem_Op<"initStrFromInt",  [AlwaysBound,
  NoCaptures]> {
  let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory,
    DynamicType:$integer);
  let results = (outs DynamicType:$result);

  let assemblyFormat = "$memory `to` $integer attr-dict";
}

def PylirMem_InitFuncOp : PylirMem_Op<"initFunc", [AlwaysBound, NoCaptures]> {
  let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory,
    FlatSymbolRefAttr:$initializer);
  let results = (outs DynamicType:$result);

  let assemblyFormat = "$memory `to` $initializer attr-dict";
}

def PylirMem_InitObjectOp : PylirMem_Op<"initObject", [AlwaysBound,
  NoCaptures]> {
  let arguments = (ins Arg<MemoryType, "", [MemWrite]>:$memory);
  let results = (outs DynamicType:$result);

  let assemblyFormat = "$memory attr-dict";
}

#endif
