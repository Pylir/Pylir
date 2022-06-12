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
include "mlir/Interfaces/CallInterfaces.td"
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

def TypeFlow_UndefAttr : TypeFlow_Attr<"Undef"> {
	let mnemonic = "undef";
}

def DictionaryArray : TypedArrayAttrBase<DictionaryAttr, "Dictionary array attribute">;

def TypeFlow_FuncOp : TypeFlow_Op<"func", [Symbol, FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface]> {
	let arguments = (ins StrAttr:$sym_name, TypeAttrOf<FunctionType>:$function_type,
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

def TypeFlow_UndefOp : TypeFlow_Op<"undef", [NoSideEffect, ConstantLike]> {
	let arguments = (ins);
	let results = (outs TypeFlow_MetaType:$result);

	let assemblyFormat = "attr-dict";

	let hasFolder = 1;
}

def TypeFlow_ConstantOp : TypeFlow_Op<"constant", [NoSideEffect, ConstantLike]> {
	let arguments = (ins AnyAttr:$input);
	let results = (outs TypeFlow_MetaType:$result);

	let assemblyFormat = "$input attr-dict";

	let hasFolder = 1;
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

def TypeFlow_CalcOp : TypeFlow_Op<"calc", [DeclareOpInterfaceMethods<InferTypeOpInterface>, NoSideEffect]> {
	let arguments = (ins Variadic<TypeFlow_MetaType>:$input,
	                     TypeFlow_InstructionAttr:$instruction,
	                     UnitAttr:$value_calc);
	let results = (outs Variadic<TypeFlow_MetaType>:$results);

	let assemblyFormat = "(`value` $value_calc^)? $input attr-dict $instruction";
}

def TypeFlow_BranchOp : TypeFlow_Op<"branch",
	[Terminator, DeclareOpInterfaceMethods<BranchOpInterface>, NoSideEffect]> {
	let arguments = (ins VariadicOfVariadic<TypeFlow_MetaType, "arg_sizes">:$branch_args, I32ElementsAttr:$arg_sizes);
	let results = (outs);

	let successors = (successor VariadicSuccessor<AnySuccessor>:$successors);

	let assemblyFormat = "$successors (`,` $branch_args^)? attr-dict";
}

def TypeFlow_CondBranchOp : TypeFlow_Op<"cond_br", [Terminator, NoSideEffect,
	DeclareOpInterfaceMethods<BranchOpInterface>, AttrSizedOperandSegments]> {
	let arguments = (ins TypeFlow_MetaType:$condition,
						 Variadic<TypeFlow_MetaType>:$true_args,
						 Variadic<TypeFlow_MetaType>:$false_args);
	let results = (outs);

	let successors = (successor AnySuccessor:$true_succ, AnySuccessor:$false_succ);

	let assemblyFormat = [{
		$condition `,` $true_succ (`(` $true_args^ `)`)? $false_succ (`(` $false_args^ `)`)? attr-dict
	}];
}

def TypeFlow_LoopRegionOp : TypeFlow_Op<"loop", [Terminator, RecursiveSideEffects]> {
	let arguments = (ins Variadic<TypeFlow_MetaType>:$region_init);
	let results = (outs);

	let description = [{
		Region representing a natural loop. Evaluation of the loop has to converge before any successors of the loop
		region may be evaluated. The loop may only do another iteration if during evaluation, a `yield` op was reached.
		In the case that a `yield` op is continuously reached, the loop will automatically exit if the block arguments
		of either the body reached a fixpoint.

		The loops `region_init` argument will be passed to the bodys block arguments at first execution. Following
		executions will get the arguments passed from the `yield` op.

		The loop region has multiple successor blocks. During evaluation of the loop header and body, `exit` ops may
		mark a successor as reachable, via their index. After the loop has exited, any reachable successors will be
		evaluated.
	}];

	let regions = (region MinSizedRegion<1>:$body);
	let successors = (successor VariadicSuccessor<AnySuccessor>:$successors);

	let assemblyFormat = "$region_init $body `successors` $successors attr-dict";
}

def TypeFlow_YieldOp : TypeFlow_Op<"yield", [Terminator, ReturnLike, NoSideEffect, HasParent<"LoopRegionOp">]> {
	let arguments = (ins Variadic<TypeFlow_MetaType>:$args);

	let description = [{
		This op may only appear in the body region of a `loop` op. It causes the possible start of the next loop
		iteration if the conditions documented in `loop` op are met.
		If the loop has a header region, `args` are forwarded to the block arguments of that region. Otherise, they are
		forwarded to the block arguments of the body region.
	}];

	let assemblyFormat = "$args attr-dict";
}

def TypeFlow_ExitOp : TypeFlow_Op<"exit", [Terminator, ReturnLike, NoSideEffect, HasParent<"LoopRegionOp">]> {
	let arguments = (ins I32Attr:$exit_succ, Variadic<TypeFlow_MetaType>:$args);

	let description = [{
		This op represents an exit out of the loop. `exit_succ` is an index into the loop ops successor, which is
		marked as reachable with the specified `args` as block arguments by the `exit` op.
	}];

	let assemblyFormat = "$exit_succ (`(` $args^ `)`)? attr-dict";
}

def TypeFlow_ReturnOp : TypeFlow_Op<"return", [Terminator, ReturnLike, NoSideEffect]> {
	let arguments = (ins Variadic<TypeFlow_MetaType>:$values);
	let results = (outs);

    let assemblyFormat = "$values attr-dict";
}


class TypeFlow_CallBase<string name, Constraint calleeType> : TypeFlow_Op<name,
	[DeclareOpInterfaceMethods<CallOpInterface>, NoSideEffect]> {
	let arguments = (ins calleeType:$callee, Variadic<TypeFlow_MetaType>:$arguments);
	let results = (outs Variadic<TypeFlow_MetaType>:$results);

	let assemblyFormat = "$callee `(`$arguments `)` `:` type(results) attr-dict";
}

def TypeFlow_CallOp : TypeFlow_CallBase<"call", SymbolRefAttr>;
def TypeFlow_CallIndirectOp : TypeFlow_CallBase<"call_indirect", TypeFlow_MetaType>;

#endif
