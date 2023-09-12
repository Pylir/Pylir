// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_MEMORY_SSA_TABLEGEN
#define PYLIR_MEMORY_SSA_TABLEGEN

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def MemorySSA_Dialect : Dialect {
  let name = "memSSA";
  let cppNamespace = "::pylir::MemSSA";

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

class MemorySSA_Type<string name, list<Trait> traits = [],
    string baseCppClass = "::mlir::Type">
  : TypeDef<MemorySSA_Dialect, name, traits, baseCppClass>;

class MemorySSA_Attr<string name, list<Trait> traits = [],
    string baseCppClass = "::mlir::Attribute">
  : AttrDef<MemorySSA_Dialect, name, traits, baseCppClass>;

class MemorySSA_Op<string mnemonic, list<Trait> traits = []>
  : Op<MemorySSA_Dialect, mnemonic, traits>;

def MemoryDefType : MemorySSA_Type<"Def"> {
  let mnemonic = "def";
}

def InstrAttr : MemorySSA_Attr<"Instruction"> {
  let mnemonic = "instruction";
  let parameters = (ins "::mlir::Operation*":$instruction);
  let returnType = "::mlir::Operation*";
  let convertFromStorage = "$_self.getInstruction()";

  let constBuilderCall = [{
    ::pylir::MemSSA::InstructionAttr::get($_builder.getContext(), $0)
  }];
  let hasCustomAssemblyFormat = 1;
}

def ReadWriteAttr : MemorySSA_Attr<"ReadWrite"> {
  let mnemonic = "readWrite";
  let parameters = (ins
    ArrayRefParameter<"llvm::PointerUnion<mlir::Value, mlir::SymbolRefAttr>">:
      $locations);
  let returnType = [{
    ::llvm::ArrayRef<::llvm::PointerUnion<::mlir::Value, ::mlir::SymbolRefAttr>>
  }];
  let convertFromStorage = "$_self.getLocations()";

  let constBuilderCall = [{
    ::pylir::MemSSA::ReadWriteAttr::get($_builder.getContext(), $0)
  }];
  let hasCustomAssemblyFormat = 1;
}

def MemoryModuleOp : MemorySSA_Op<"module", [IsolatedFromAbove,
  DeclareOpInterfaceMethods<RegionKindInterface>,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getDefaultDialect"]>]> {
  let arguments = (ins);
  let results = (outs);

  let regions = (region AnyRegion:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}

def MemoryLiveOnEntryOp : MemorySSA_Op<"liveOnEntry"> {
  let arguments = (ins );
  let results = (outs MemoryDefType:$result);

  let assemblyFormat = "attr-dict";
}

def MemoryUseOp : MemorySSA_Op<"use"> {
  let arguments = (ins
    MemoryDefType:$definition,
    InstrAttr:$instruction,
    ReadWriteAttr:$reads
  );
  let results = (outs);

  let assemblyFormat = [{
    `(` $definition `)` attr-dict `\n` $instruction $reads
  }];
}

def MemoryDefOp : MemorySSA_Op<"def"> {
  let arguments = (ins
    MemoryDefType:$clobbered,
    InstrAttr:$instruction,
    ReadWriteAttr:$writes,
    ReadWriteAttr:$reads
  );
  let results = (outs MemoryDefType:$result);

  let assemblyFormat = [{
    `(` $clobbered `)` attr-dict `\n` $instruction $writes $reads
  }];
}

def MemoryBranchOp : MemorySSA_Op<"br", [Terminator,
  DeclareOpInterfaceMethods<BranchOpInterface>]> {
  let arguments = (ins
    VariadicOfVariadic<MemoryDefType, "arg_sizes">:$branch_args,
    DenseI32ArrayAttr:$arg_sizes
  );
  let results = (outs);

  let successors = (successor VariadicSuccessor<AnySuccessor>:$successors);

  let assemblyFormat = "$successors $branch_args attr-dict";

  let hasVerifier = 1;
}

#endif
