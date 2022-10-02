// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_PY_TYPES_TABLEGEN
#define PYLIR_PY_TYPES_TABLEGEN

include "mlir/IR/SubElementInterfaces.td"
include "pylir/Optimizer/PylirPy/IR/PylirPyBase.td"
include "pylir/Optimizer/Util/TablegenUtil.td"
include "pylir/Optimizer/PylirPy/IR/ObjectTypeInterface.td"

class PylirPy_Type<string name, list<Trait> traits = []> : TypeDef<PylirPy_Dialect, name, traits>;

class PylirPy_ObjectType<string name, list<Trait> traits = []>
	: PylirPy_Type<name, !listconcat(traits, [ObjectTypeInterface])>;

def PylirPy_UnboundType : PylirPy_ObjectType<"Unbound"> {
    let mnemonic = "unbound";
    let summary = "unbound type";

    let extraClassDeclaration = [{
        pylir::Py::RefAttr getTypeObject() const
        {
            return {};
        }
    }];
}

def PylirPy_ClassType : PylirPy_ObjectType<"Class", [DeclareTypeInterfaceMethods<SubElementTypeInterface>]> {
	let mnemonic = "class";
	let summary = "class type";

	let parameters = (ins "RefAttr":$typeObject);

	let builders = [TypeBuilderWithInferredContext<parameters, [{
		return get(typeObject.getContext(), typeObject);
	}]>];

	let assemblyFormat = "`<` $typeObject `>`";
}

def PylirPy_TupleType : PylirPy_ObjectType<"Tuple", [DeclareTypeInterfaceMethods<SubElementTypeInterface>]> {
	let mnemonic = "tuple";
	let summary = "tuple type";

	let parameters = (ins DefaultType<"Tuple">:$typeObject, ListPrint<"pylir::Py::ObjectTypeInterface", "Paren">:$elements);

	let skipDefaultBuilders = 1;

	let builders = [TypeBuilderWithInferredContext<(ins "RefAttr":$typeObject,
										CArg<"::llvm::ArrayRef<::pylir::Py::ObjectTypeInterface>", "{}">:$elements), [{
    	return Base::get(typeObject.getContext(), typeObject, elements);
	}]>, TypeBuilder<(ins CArg<"RefAttr", "{}">:$typeObject,
						  CArg<"::llvm::ArrayRef<::pylir::Py::ObjectTypeInterface>", "{}">:$elements), [{
		typeObject = typeObject ? typeObject : }] # DefaultType<"Tuple">.defaultValue # [{;
		return get(typeObject, elements);
	}]>];

	let assemblyFormat = "`<` $elements (`,` $typeObject^)? `>`";
}

def PylirPy_VariantType : PylirPy_ObjectType<"Variant", [DeclareTypeInterfaceMethods<SubElementTypeInterface>]> {
	let mnemonic = "variant";
	let summary = "variant type";

	let parameters = (ins ArrayRefParameter<"pylir::Py::ObjectTypeInterface">:$elements);

	let assemblyFormat = "`<` params `>`";

	let extraClassDeclaration = [{

		RefAttr getTypeObject() const
        {
            return {};
		}
	}];
}

def DynamicType : PylirPy_Type<"Dynamic"> {
	let mnemonic = "dynamic";
	let summary = "dynamic type";
}

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
