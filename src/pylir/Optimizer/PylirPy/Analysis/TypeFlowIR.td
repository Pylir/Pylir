// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_TYPE_FLOW_TABLEGEN
#define PYLIR_TYPE_FLOW_TABLEGEN

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "pylir/Optimizer/PylirPy/Analysis/TypeFlowInterfaces.td"

def TypeFlow_Dialect : Dialect {
    let name = "typeFlow";
    let cppNamespace = "::pylir::TypeFlow";

    let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
    let useDefaultAttributePrinterParser = 1;
    let useDefaultTypePrinterParser = 1;
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

def TypeFlow_UndefOp : TypeFlow_Op<"undef", [NoMemoryEffect]> {
	let arguments = (ins);
	let results = (outs TypeFlow_MetaType:$result);

	let assemblyFormat = "attr-dict";
}

def TypeFlow_ConstantOp : TypeFlow_Op<"constant", [NoMemoryEffect, ConstantLike]> {
	let arguments = (ins AnyAttr:$input);
	let results = (outs TypeFlow_MetaType:$result);

	let assemblyFormat = "$input attr-dict";

	let hasFolder = 1;
}

defvar foldAsExec = [{
mlir::OpFoldResult $cppClass::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    llvm::SmallVector<::pylir::TypeFlow::OpFoldResult> res;
    if (mlir::failed(exec(llvm::to_vector(llvm::map_range(operands, [](mlir::Attribute attr) -> ::pylir::Py::TypeAttrUnion {
        if (auto typeAttr = attr.dyn_cast_or_null<mlir::TypeAttr>())
        {
            return typeAttr.getValue().cast<pylir::Py::ObjectTypeInterface>();
        }
        else
        {
            return attr;
        }
    })), res)))
    {
        return nullptr;
    }
    auto first = res.front();
    if (auto typeFlowValue = first.dyn_cast<::pylir::Py::TypeAttrUnion>())
    {
        if (auto type = typeFlowValue.dyn_cast_or_null<pylir::Py::ObjectTypeInterface>())
        {
            return mlir::TypeAttr::get(type);
        }
        return typeFlowValue.dyn_cast<mlir::Attribute>();
    }
    return first.get<mlir::Value>();
}
}];

def TypeFlow_TypeOfOp : TypeFlow_Op<"typeOf", [NoMemoryEffect, DeclareOpInterfaceMethods<TypeFlowExecInterface>,
	DeclareOpInterfaceMethods<TypeFlowValueMappingInterface>]> {
	let arguments = (ins TypeFlow_MetaType:$input, TypeFlow_InstructionAttr:$instruction);
	let results = (outs TypeFlow_MetaType:$result);

	let assemblyFormat = "$input attr-dict $instruction";

	let hasFolder = 1;

	let extraClassDefinition = foldAsExec;
}

def TypeFlow_TupleLenOp : TypeFlow_Op<"tupleLen", [NoMemoryEffect, DeclareOpInterfaceMethods<TypeFlowExecInterface>,
	DeclareOpInterfaceMethods<TypeFlowValueMappingInterface>]> {
	let arguments = (ins TypeFlow_MetaType:$input, TypeFlow_InstructionAttr:$instruction);
	let results = (outs TypeFlow_MetaType:$result);

	let assemblyFormat = "$input attr-dict $instruction";

	let hasFolder = 1;

	let extraClassDefinition = foldAsExec;
}

def TypeFlow_IsOp : TypeFlow_Op<"is", [NoMemoryEffect, DeclareOpInterfaceMethods<TypeFlowExecInterface>,
	DeclareOpInterfaceMethods<TypeFlowValueMappingInterface>]> {
	let arguments = (ins TypeFlow_MetaType:$lhs, TypeFlow_MetaType:$rhs, TypeFlow_InstructionAttr:$instruction);
	let results = (outs TypeFlow_MetaType:$result);

	let assemblyFormat = "$lhs `,` $rhs attr-dict $instruction";

	let hasFolder = 1;

	let extraClassDefinition = foldAsExec;
}

def TypeFlow_CalcOp : TypeFlow_Op<"calc", [DeclareOpInterfaceMethods<InferTypeOpInterface>, NoMemoryEffect,
	DeclareOpInterfaceMethods<TypeFlowExecInterface>, DeclareOpInterfaceMethods<TypeFlowValueMappingInterface>]> {
	let arguments = (ins Variadic<TypeFlow_MetaType>:$input,
	                     TypeFlow_InstructionAttr:$instruction,
	                     UnitAttr:$value_calc);
	let results = (outs Variadic<TypeFlow_MetaType>:$results);

	let assemblyFormat = "(`value` $value_calc^)? $input attr-dict $instruction";

	let hasFolder = 1;

	let extraClassDeclaration = [{
        mlir::LogicalResult exec(::llvm::ArrayRef<Py::TypeAttrUnion> operands,
                                 ::llvm::SmallVectorImpl<OpFoldResult>& results,
                                 bool forFolding);
	}];
}

def TypeFlow_BranchOp : TypeFlow_Op<"branch",
	[Terminator, DeclareOpInterfaceMethods<BranchOpInterface>, NoMemoryEffect]> {
	let arguments = (ins VariadicOfVariadic<TypeFlow_MetaType, "arg_sizes">:$branch_args, DenseI32ArrayAttr:$arg_sizes);
	let results = (outs);

	let successors = (successor VariadicSuccessor<AnySuccessor>:$successors);

	let assemblyFormat = "$successors (`,` $branch_args^)? attr-dict";
}

def TypeFlow_CondBranchOp : TypeFlow_Op<"cond_br", [Terminator, NoMemoryEffect,
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

def TypeFlow_ReturnOp : TypeFlow_Op<"return", [Terminator, ReturnLike, NoMemoryEffect]> {
	let arguments = (ins Variadic<TypeFlow_MetaType>:$values);
	let results = (outs);

    let assemblyFormat = "$values attr-dict";
}


class TypeFlow_CallBase<string name, Constraint calleeType> : TypeFlow_Op<name,
	[DeclareOpInterfaceMethods<CallOpInterface>, NoMemoryEffect,
	 DeclareOpInterfaceMethods<TypeFlowValueMappingInterface>]> {
	let arguments = (ins calleeType:$callee, Variadic<TypeFlow_MetaType>:$arguments,
	                     TypeFlow_InstructionAttr:$instruction);
	let results = (outs Variadic<TypeFlow_MetaType>:$results);

	let assemblyFormat = "$callee `(`$arguments `)` `:` type(results) attr-dict $instruction";
}

def TypeFlow_CallOp : TypeFlow_CallBase<"call", SymbolRefAttr>;
def TypeFlow_CallIndirectOp : TypeFlow_CallBase<"call_indirect", TypeFlow_MetaType>;

#endif
