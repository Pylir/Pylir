// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_TYPE_FLOW_TABLEGEN
#define PYLIR_TYPE_FLOW_TABLEGEN

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TypeFlow_Dialect : Dialect {
    let name = "typeFlow";
    let cppNamespace = "::pylir::TypeFlow";

    let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

class TypeFlow_Type<string name, list<Trait> traits = [], string baseCppClass = "::mlir::Type">
    : TypeDef<TypeFlow_Dialect, name, traits, baseCppClass>;

def TypeFlow_MetaType : TypeFlow_Type<"Meta"> {
	let mnemonic = "meta";
}

class TypeFlow_Attr<string name, list<Trait> traits = [], string baseCppClass = "::mlir::Attribute">
    : AttrDef<TypeFlow_Dialect, name, traits, baseCppClass>;

class TypeFlow_Op<string mnemonic, list<Trait> traits = []>
    : Op<TypeFlow_Dialect, mnemonic, traits>;

def TypeFlow_InstructionAttr : TypeFlow_Attr<"Instruction"> {
    let mnemonic = "instruction";
    let parameters = (ins "::mlir::Operation*":$instruction);
    let returnType = "::mlir::Operation*";
    let convertFromStorage = "$_self.getInstruction()";

    let constBuilderCall = "::pylir::TypeFlow::InstructionAttr::get($_builder.getContext(), $0)";
    let hasCustomAssemblyFormat = 1;
}

def DictionaryArray : TypedArrayAttrBase<DictionaryAttr, "Dictionary array attribute">;

def TypeFlow_FuncOp : TypeFlow_Op<"func", [Symbol, FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface]> {
	let arguments = (ins TypeAttrOf<FunctionType>:$function_type,
					 DictionaryArray:$arg_attrs,
					 DictionaryArray:$res_attrs);

	let regions = (region AnyRegion:$body);

	let extraClassDeclaration = [{
        /// Returns the argument types of this function.
        llvm::ArrayRef<mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }

        /// Returns the result types of this function.
        llvm::ArrayRef<mlir::Type> getResultTypes() { return getFunctionType().getResults(); }

        static llvm::StringRef getDefaultDialect()
        {
            return pylir::TypeFlow::TypeFlowDialect::getDialectNamespace();
        }
	}];

	let hasCustomAssemblyFormat = 1;
}

def TypeFlow_TypeOfOp : TypeFlow_Op<"typeOf", [NoSideEffect]> {
	let arguments = (ins TypeFlow_MetaType:$input);
	let results = (outs TypeFlow_MetaType:$result);

	let assemblyFormat = "$input attr-dict";
}

def TypeFlow_MakeObjectOp : TypeFlow_Op<"makeObject", [NoSideEffect]> {
	let arguments = (ins TypeFlow_MetaType:$input);
	let results = (outs TypeFlow_MetaType:$result);

	let assemblyFormat = "$input attr-dict";
}

def TypeFlow_RefineableOp : TypeFlow_Op<"refineable", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
	let arguments = (ins Variadic<TypeFlow_MetaType>:$input, TypeFlow_InstructionAttr:$instruction);
	let results = (outs Variadic<TypeFlow_MetaType>:$results);

	let assemblyFormat = "$input attr-dict $instruction";
}

def TypeFlow_FoldableOp : TypeFlow_Op<"foldable", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
	let arguments = (ins Variadic<TypeFlow_MetaType>:$input, I32ArrayAttr:$indices, TypeFlow_InstructionAttr:$instruction);
	let results = (outs Variadic<AnyType>:$results);

	let assemblyFormat = "$input $indices attr-dict $instruction";
}

def TypeFlow_BranchOp : TypeFlow_Op<"branch", [Terminator, DeclareOpInterfaceMethods<BranchOpInterface>]> {
	let arguments = (ins VariadicOfVariadic<TypeFlow_MetaType, "arg_sizes">:$branch_args, I32ElementsAttr:$arg_sizes);
	let results = (outs);

	let successors = (successor VariadicSuccessor<AnySuccessor>:$successors);

	let assemblyFormat = "$successors $branch_args attr-dict";
}

#endif
