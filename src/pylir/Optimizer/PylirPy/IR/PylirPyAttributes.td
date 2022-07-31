// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_PY_ATTR_TABLEGEN
#define PYLIR_PY_ATTR_TABLEGEN

include "pylir/Optimizer/PylirPy/IR/PylirPyBase.td"
include "pylir/Optimizer/PylirPy/IR/ObjectAttrInterface.td"
include "pylir/Optimizer/PylirPy/IR/IntAttrInterface.td"
include "pylir/Optimizer/Util/TablegenUtil.td"
include "mlir/IR/SubElementInterfaces.td"

class PylirPy_Attr<string name, list<Trait> traits = [], string baseCppClass = "::mlir::Attribute"> :
    AttrDef<PylirPy_Dialect, name, traits, baseCppClass>;

defvar SlotsMap = DefaultValuedParameter<"::mlir::DictionaryAttr", "::mlir::DictionaryAttr::get($_ctxt, {})">;

class AttrSpecBuilder<string name, list<ParamSpec> spec, list<Trait> traits = [], string extraBuilderCode = "">
	: PylirPy_Attr<name, traits> {

	let parameters = !dag(ins, !foreach(x, spec, x.type), !foreach(x, spec, x.name));

    let skipDefaultBuilders = 1;

    let builders = [ParamSpecBuilder<spec, extraBuilderCode>];
}

class TypeObjectSlotsAttr<string name, list<ParamSpec> parameters, list<Trait> traits = [], string extraBuilderCode = "">
    : AttrSpecBuilder<name,
    !listconcat(parameters, [ParamSpec<DefaultType<name>, "typeObject">, ParamSpec<SlotsMap, "slots">]),
    !listconcat(traits, [DeclareAttrInterfaceMethods<SubElementAttrInterface,
										["replaceImmediateSubAttribute"]>, ObjectAttrInterface]),
										extraBuilderCode>;


class TypeObjectNoSlotsAttr<string name, list<ParamSpec> parameters, list<Trait> traits = []>
    : AttrSpecBuilder<name,
    !listconcat(parameters, [ParamSpec<DefaultType<name>, "typeObject">]),
    !listconcat(traits, [DeclareAttrInterfaceMethods<SubElementAttrInterface,
										["replaceImmediateSubAttribute"]>, ObjectAttrInterface])>;


def PylirPy_ObjectAttr : TypeObjectSlotsAttr<"Object", [], [
	DeclareAttrInterfaceMethods<SubElementAttrInterface, ["replaceImmediateSubAttribute"]>]> {
	let mnemonic = "obj";
	let summary = "python object";

	let assemblyFormat = "`<` $typeObject (`,` $slots^)? `>`";
}

def PylirPy_IntAttr : TypeObjectSlotsAttr<"Int", [
	ParamSpec<CustomPrint<AttrParameter<"pylir::BigInt", "", "const pylir::BigInt&">,
											?, "$_printer << $_self.toString();">, "value">],
	[DeclareAttrInterfaceMethods<IntAttrInterface>]> {
	let mnemonic = "int";
	let summary = "python integer";

	let assemblyFormat = "`<` $value ( `,` struct($typeObject, $slots)^)? `>`";

	let constBuilderCall = "::pylir::Py::IntAttr::get($_builder.getContext(), $0)";
}

def PylirPy_BoolAttr : TypeObjectSlotsAttr<"Bool", [ParamSpec<CustomPrint<AttrParameter<"bool", "">, [{
	[&]() -> mlir::FailureOr<bool> {
		llvm::StringRef keyword;
        auto loc = $_parser.getCurrentLocation();
        if ($_parser.parseKeyword(&keyword))
        {
            return mlir::failure();
        }
        if (keyword != "True" && keyword != "False")
        {
            $_parser.emitError(loc, "Expected 'True' or 'False' instead of ") << keyword;
            return mlir::failure();
        }
        return keyword == "True";
	}()
}], [{
	$_printer << ($_self ? "True" : "False");
}]>, "value">],
	[DeclareAttrInterfaceMethods<IntAttrInterface>]> {
	let mnemonic = "bool";
	let summary = "python boolean";

	let assemblyFormat = "`<`$value ( `,` struct($typeObject, $slots)^)?`>`";
}

def PylirPy_FloatAttr : TypeObjectSlotsAttr<"Float", [ParamSpec<APFloatParameter<"">, "value">]> {
	let mnemonic = "float";
	let summary = "python float";

    let extraClassDeclaration = [{
        double getDoubleValue() const
        {
            return getValue().convertToDouble();
        }
    }];

    let assemblyFormat = "`<` $value ( `,` struct($typeObject, $slots)^)? `>`";
}

def PylirPy_StrAttr : TypeObjectSlotsAttr<"Str", [ParamSpec<CustomPrint<StringRefParameter<>, ?, [{
	$_printer << "\"";
	llvm::printEscapedString($_self, $_printer.getStream());
	$_printer << "\"";
}]>, "value">]> {
	let mnemonic = "str";
	let summary = "python string";

	let assemblyFormat = "`<` $value ( `,` struct($typeObject, $slots)^)? `>`";
}

def PylirPy_TupleAttr : TypeObjectNoSlotsAttr<"Tuple",
	[ParamSpec<ListPrint<"::mlir::Attribute", "Paren">, "value", "{}">]> {
	let mnemonic = "tuple";
	let summary = "python tuple";

	let extraClassDeclaration = [{
		::mlir::DictionaryAttr getSlots() const
    	{
            return ::mlir::DictionaryAttr::getWithSorted(getContext(), {});
		}
	}];

	let assemblyFormat = "`<` $value ( `,` struct($typeObject)^)? `>`";
}

def PylirPy_ListAttr : TypeObjectSlotsAttr<"List",
	[ParamSpec<ListPrint<"::mlir::Attribute", "Square">, "value", "{}">]> {
	let mnemonic = "list";

	let assemblyFormat = "`<` $value ( `,` struct($typeObject, $slots)^)? `>`";
}

def PylirPy_SetAttr : TypeObjectSlotsAttr<"Set",
	[ParamSpec<ListPrint<"::mlir::Attribute", "Braces">, "value", "{}">], [], [{
		std::vector<mlir::Attribute> temp = value.vec();
        temp.erase(std::unique(temp.begin(), temp.end()), temp.end());
		value = temp;
	}]> {
	let mnemonic = "set";
	let summary = "python set";

	let assemblyFormat = "`<` $value ( `,` struct($typeObject, $slots)^)? `>`";
}

def PylirPy_DictAttr : TypeObjectSlotsAttr<"Dict",
	[ParamSpec<ListPrint<"std::pair<mlir::Attribute, mlir::Attribute>", "Braces", [{
		[&]() -> mlir::FailureOr<std::pair<mlir::Attribute, mlir::Attribute>> {
			mlir::Attribute first, second;
			if ($_parser.parseAttribute(first) || $_parser.parseKeyword("to") || $_parser.parseAttribute(second))
			{
				return mlir::failure();
			}
			return std::pair{first, second};
		}()
	}], [{
		$_printer << $_self.first << " to " << $_self.second;
	}]>, "value", "{}">], [], [{
		auto vector = value.vec();
		vector.erase(std::unique(vector.begin(), vector.end(),
                        [](const auto& lhs, const auto& rhs) { return lhs.first == rhs.first; }), vector.end());
		value = vector;
	}]> {
    let mnemonic = "dict";
    let summary = "python dictionary";

    let assemblyFormat = "`<` $value ( `,` struct($typeObject, $slots)^)? `>`";
}

def PylirPy_FunctionAttr : AttrSpecBuilder<"Function",
	[ParamSpec<AttrParameter<"mlir::FlatSymbolRefAttr", "">, "value">,
	 ParamSpec<DefaultValuedParameter<"mlir::Attribute", "::pylir::Py::StrAttr::get($_ctxt, \"\")">, "qualName">,
	 ParamSpec<DefaultValuedParameter<"mlir::Attribute", "::mlir::FlatSymbolRefAttr::get($_ctxt, pylir::Builtins::None.name)">, "defaults">,
	 ParamSpec<DefaultValuedParameter<"mlir::Attribute", "::mlir::FlatSymbolRefAttr::get($_ctxt, pylir::Builtins::None.name)">, "kwDefaults">,
	 ParamSpec<DefaultValuedParameter<"mlir::Attribute", "::mlir::Attribute{}">, "dict">
	 ], [DeclareAttrInterfaceMethods<SubElementAttrInterface, ["replaceImmediateSubAttribute"]>,
	     DeclareAttrInterfaceMethods<ObjectAttrInterface>]>
{
	let mnemonic = "function";
	let summary = "python function";

	let assemblyFormat = "`<` $value ( `,` struct($qualName, $defaults, $kwDefaults, $dict)^)? `>`";
}

def PylirPy_TypeAttr : TypeObjectSlotsAttr<"Type",
	[ParamSpec<DefaultValuedParameter<"::mlir::Attribute", "::pylir::Py::TupleAttr::get($_ctxt, {})">, "mroTuple">]> {
	let mnemonic = "type";
	let summary = "python type";

	let assemblyFormat = "(`<`struct(params)^`>`)?";
}

def PylirPy_UnboundAttr : PylirPy_Attr<"Unbound"> {
    let mnemonic = "unbound";
    let summary = "python unbound value";
}

def AnyPylirAttr : AnyAttrOf<[PylirPy_UnboundAttr, SymbolRefAttr, ObjectAttrInterface]> {
	let valueType = NoneType;
}

#endif
