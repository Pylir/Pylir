
#ifndef PYLIR_MEMORY_SSA_TABLEGEN
#define PYLIR_MEMORY_SSA_TABLEGEN

include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def MemorySSA_Dialect : Dialect {
    let name = "memSSA";
    let cppNamespace = "::pylir::MemSSA";
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
}

class MemorySSA_Type<string name, list<Trait> traits = [], string baseCppClass = "::mlir::Type">
    : TypeDef<MemorySSA_Dialect, name, traits, baseCppClass>;

class MemorySSA_Attr<string name, list<Trait> traits = [], string baseCppClass = "::mlir::Attribute">
    : AttrDef<MemorySSA_Dialect, name, traits, baseCppClass>;

class MemorySSA_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<MemorySSA_Dialect, mnemonic, traits>;

def MemoryDefType : MemorySSA_Type<"Def"> {
    let mnemonic = "def";
}

def InstrAttr : MemorySSA_Attr<"Instruction"> {
    let mnemonic = "instruction";
    let parameters = (ins "::mlir::Operation*":$instruction);
    let returnType = "::mlir::Operation*";
    let convertFromStorage = "$_self.getInstruction()";

    let constBuilderCall = "::pylir::MemSSA::InstructionAttr::get($_builder.getContext(), $0)";
}

def ReadAttr : MemorySSA_Attr<"Read"> {
    let mnemonic = "read";
    let parameters = (ins "::mlir::Value":$read);
    let returnType = "::mlir::Value";
    let convertFromStorage = "$_self.getRead()";

    let constBuilderCall = "::pylir::MemSSA::ReadAttr::get($_builder.getContext(), $0)";
}

def MemoryRegionOp : MemorySSA_Op<"region", [IsolatedFromAbove, DeclareOpInterfaceMethods<RegionKindInterface>]> {
    let arguments = (ins FlatSymbolRefAttr:$functionName);
    let results = (outs);

    let regions = (region AnyRegion:$body);
    let assemblyFormat = "$functionName attr-dict-with-keyword $body";
}

def MemoryLiveOnEntryOp : MemorySSA_Op<"liveOnEntry"> {
    let arguments = (ins );
    let results = (outs MemoryDefType:$result);

    let assemblyFormat = "attr-dict";
}

def MemoryUseOp : MemorySSA_Op<"use"> {
    let arguments = (ins MemoryDefType:$definition, InstrAttr:$instruction, ReadAttr:$read);
    let results = (outs);

    let assemblyFormat = "`(` $definition `)` attr-dict `\\n` $instruction $read";
}

def MemoryDefOp : MemorySSA_Op<"def"> {
    let arguments = (ins MemoryDefType:$clobbered, InstrAttr:$instruction);
    let results = (outs MemoryDefType:$result);

    let assemblyFormat = "`(` $clobbered `)` attr-dict `\\n` $instruction";
}

def MemoryBranchOp : MemorySSA_Op<"br", [Terminator, DeclareOpInterfaceMethods<BranchOpInterface>]> {
    let arguments = (ins VariadicOfVariadic<MemoryDefType, "argSizes">:$branchArgs, I32ElementsAttr:$argSizes);
    let results = (outs);

    let successors = (successor VariadicSuccessor<AnySuccessor>:$successors);

    let assemblyFormat = "$successors $branchArgs attr-dict";

    let verifier = [{
        if (branchArgs().size() != successors().size())
        {
            return emitOpError("Expected branch arguments for every successor");
        }
        return mlir::success();
    }];
}

#endif
