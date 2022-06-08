// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_PY_TYPES_TABLEGEN
#define PYLIR_PY_TYPES_TABLEGEN

include "pylir/Optimizer/PylirPy/IR/PylirPyBase.td"
include "pylir/Optimizer/Util/TablegenUtil.td"
include "pylir/Optimizer/PylirPy/IR/ObjectTypeInterface.td"

class PylirPy_Type<string name, list<Trait> traits = []> : TypeDef<PylirPy_Dialect, name, traits>;

class PylirPy_ObjectType<string name, list<Trait> traits = []>
	: PylirPy_Type<name, !listconcat(traits, [ObjectTypeInterface])>;

def PylirPy_UnknownType : PylirPy_ObjectType<"Unknown"> {
    let mnemonic = "unknown";
    let summary = "unknown type";

    let extraClassDeclaration = [{
        mlir::FlatSymbolRefAttr getTypeObject() const
        {
            return {};
        }
    }];
}

def PylirPy_UnboundType : PylirPy_ObjectType<"Unbound"> {
    let mnemonic = "unbound";
    let summary = "unbound type";

    let extraClassDeclaration = [{
        mlir::FlatSymbolRefAttr getTypeObject() const
        {
            return {};
        }
    }];
}

def PylirPy_ClassType : PylirPy_ObjectType<"Class"> {
	let mnemonic = "class";
	let summary = "class type";

	let parameters = (ins "::mlir::FlatSymbolRefAttr":$typeObject);

	let builders = [TypeBuilderWithInferredContext<parameters, [{
		return get(typeObject.getContext(), typeObject);
	}]>];

	let assemblyFormat = "`<` $typeObject `>`";
}

def PylirPy_TupleType : PylirPy_ObjectType<"Tuple"> {
	let mnemonic = "tuple";
	let summary = "tuple type";

	let parameters = (ins DefaultType<"Tuple">:$typeObject, ListPrint<"pylir::Py::ObjectTypeInterface", "Paren">:$elements);

	let builders = [TypeBuilderWithInferredContext<(ins "::mlir::FlatSymbolRefAttr":$typeObject,
													"::llvm::ArrayRef<::pylir::Py::ObjectTypeInterface>":$elements), [{
    	return get(typeObject.getContext(), typeObject, elements);
	}]>];

	let assemblyFormat = "`<` $elements (`,` $typeObject^)? `>`";
}

def PylirPy_VariantType : PylirPy_ObjectType<"Variant"> {
	let mnemonic = "variant";
	let summary = "variant type";

	let parameters = (ins ArrayRefParameter<"pylir::Py::ObjectTypeInterface">:$elements);

	let assemblyFormat = "`<` params `>`";

	let extraClassDeclaration = [{

		mlir::FlatSymbolRefAttr getTypeObject() const
        {
            return {};
		}
	}];
}

def DynamicType : PylirPy_Type<"Dynamic"> {
	let mnemonic = "dynamic";
	let summary = "dynamic type";
}

defvar IntegerLike = TypeConstraint<Or<[Index.predicate, AnySignlessInteger.predicate]>>;
def PyFunctionType : Type<
    And<[
        FunctionType.predicate,
        CPred<"$_self.cast<::mlir::FunctionType>().getNumResults() == 1">,
        SubstLeaves<"$_self", "$_self.cast<::mlir::FunctionType>().getResult(0)", DynamicType.predicate>,
        CPred<"$_self.cast<::mlir::FunctionType>().getNumInputs() == 3">,
        SubstLeaves<"$_self", "$_self.cast<::mlir::FunctionType>().getInput(0)", DynamicType.predicate>,
        SubstLeaves<"$_self", "$_self.cast<::mlir::FunctionType>().getInput(1)", DynamicType.predicate>,
        SubstLeaves<"$_self", "$_self.cast<::mlir::FunctionType>().getInput(2)", DynamicType.predicate>
    ]>>
{
    let summary = "python function";
}

#endif
