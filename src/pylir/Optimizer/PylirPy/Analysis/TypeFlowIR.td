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

def TypeFlow_Dialect : Dialect {
    let name = "typeFlow";
    let cppNamespace = "::pylir::TypeFlow";

    let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

class TypeFlow_Type<string name, list<Trait> traits = [], string baseCppClass = "::mlir::Type">
    : TypeDef<TypeFlow_Dialect, name, traits, baseCppClass>;

class TypeFlow_Attr<string name, list<Trait> traits = [], string baseCppClass = "::mlir::Attribute">
    : AttrDef<TypeFlow_Dialect, name, traits, baseCppClass>;

class TypeFlow_Op<string mnemonic, list<Trait> traits = []>
    : Op<TypeFlow_Dialect, mnemonic, traits>;

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

#endif
