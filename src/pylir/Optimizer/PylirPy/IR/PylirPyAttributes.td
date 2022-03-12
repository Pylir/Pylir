
#ifndef PYLIR_PY_ATTR_TABLEGEN
#define PYLIR_PY_ATTR_TABLEGEN

include "pylir/Optimizer/PylirPy/IR/PylirPyBase.td"
include "pylir/Optimizer/PylirPy/IR/ObjectAttrInterface.td"
include "pylir/Optimizer/PylirPy/IR/IntAttrInterface.td"
include "mlir/IR/SubElementInterfaces.td"

class PylirPy_Attr<string name, list<Trait> traits = [], string baseCppClass = "::mlir::Attribute"> :
    AttrDef<PylirPy_Dialect, name, traits, baseCppClass>;

defvar SlotsMap = DefaultValuedParameter<"::mlir::DictionaryAttr", "::mlir::DictionaryAttr::get($_ctx, {})">;

def PylirPy_ObjectAttr : PylirPy_Attr<"Object", [
	DeclareAttrInterfaceMethods<SubElementAttrInterface, ["replaceImmediateSubAttribute"]>, ObjectAttrInterface]> {
	let mnemonic = "obj";

	let parameters = (ins "::mlir::FlatSymbolRefAttr":$typeObject,
						  SlotsMap:$slots,
						  OptionalParameter<"::mlir::Attribute">:$builtinValue);

	let builders = [
		AttrBuilderWithInferredContext<(ins "::mlir::FlatSymbolRefAttr":$typeObject,
											CArg<"::mlir::DictionaryAttr", "{}">:$slots,
											CArg<"::mlir::Attribute", "{}">:$builtinValue), [{
			return Base::get(typeObject.getContext(), typeObject,
						slots ? slots : mlir::DictionaryAttr::getWithSorted(typeObject.getContext(), {}), builtinValue);
		}]>
	];

	let assemblyFormat = "`<`struct(params)`>`";
}

class DefaultType<string name, string description = ""> : DefaultValuedParameter<"::mlir::FlatSymbolRefAttr",
	"::mlir::FlatSymbolRefAttr::get($_ctx, ::pylir::Py::Builtins::" # name # ".name)", description>;

class ParamSpec<AttrOrTypeParameter m_type, string m_name, string m_defaultArg = ""> {
	AttrOrTypeParameter type = m_type;
	string name = m_name;
	string defaultArg = m_defaultArg;
}

class ParamSpecBuilder<list<ParamSpec> spec, code extraCode = ""> : AttrBuilder<!dag(ins,
											!foreach(x, spec, CArg<x.type.cppType, !if(x.type.isOptional, "{}", x.defaultArg)>),
															!foreach(x, spec, x.name)),
	!interleave(!foreach(x, !filter(x, spec, x.type.isOptional),
					x.name # " = " # x.name # " ? " # x.name # " : " # !subst("$_ctx", "$_ctxt", x.type.defaultValue) # ";\n"),
							"") # extraCode #
	"return Base::get($_ctxt, " # !interleave(!foreach(x, spec, x.name), ", ") # ");"
>;

class ParamSpecDag<list<ParamSpec> spec> {
	dag result = !dag(ins, !foreach(x, spec, x.type), !foreach(x, spec, x.name));
}

class AttrSpecBuilder<string name, list<ParamSpec> spec, list<Trait> traits = [], string extraBuilderCode = "">
	: PylirPy_Attr<name, traits> {

	let parameters = ParamSpecDag<spec>.result;

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

def PylirPy_IntAttr : TypeObjectSlotsAttr<"Int", [
	ParamSpec<AttrParameter<"pylir::BigInt", "", "const pylir::BigInt&">, "value">],
	[DeclareAttrInterfaceMethods<IntAttrInterface>]> {
	let mnemonic = "int";
}

def PylirPy_BoolAttr : TypeObjectSlotsAttr<"Bool", [ParamSpec<AttrParameter<"bool", "">, "value">],
	[DeclareAttrInterfaceMethods<IntAttrInterface>]> {
	let mnemonic = "bool";
}

def PylirPy_FloatAttr : TypeObjectSlotsAttr<"Float", [ParamSpec<APFloatParameter<"">, "value">]> {
	let mnemonic = "float";

    let extraClassDeclaration = [{
        double getDoubleValue() const
        {
            return getValue().convertToDouble();
        }
    }];
}

def PylirPy_StrAttr : TypeObjectSlotsAttr<"Str", [ParamSpec<StringRefParameter<>, "value">]> {
	let mnemonic = "str";
}

def PylirPy_TupleAttr : TypeObjectNoSlotsAttr<"Tuple",
	[ParamSpec<ArrayRefParameter<"::mlir::Attribute">, "value", "{}">]> {
	let mnemonic = "tuple";

	let extraClassDeclaration = [{
		::mlir::DictionaryAttr getSlots() const
    	{
            return ::mlir::DictionaryAttr::getWithSorted(getContext(), {});
		}
	}];
}

def PylirPy_ListAttr : TypeObjectSlotsAttr<"List",
	[ParamSpec<ArrayRefParameter<"::mlir::Attribute">, "value", "{}">]> {
	let mnemonic = "list";
}

def PylirPy_SetAttr : TypeObjectSlotsAttr<"Set",
	[ParamSpec<ArrayRefParameter<"::mlir::Attribute">, "value", "{}">], [], [{
		std::vector<mlir::Attribute> temp = value.vec();
        temp.erase(std::unique(temp.begin(), temp.end()), temp.end());
		value = temp;
	}]> {
	let mnemonic = "set";
}

def PylirPy_DictAttr : TypeObjectSlotsAttr<"Dict",
	[ParamSpec<ArrayRefParameter<"std::pair<mlir::Attribute, mlir::Attribute>">, "value", "{}">], [], [{
		auto vector = value.vec();
		vector.erase(std::unique(vector.begin(), vector.end(),
                        [](const auto& lhs, const auto& rhs) { return lhs.first == rhs.first; }), vector.end());
		value = vector;
	}]> {
    let mnemonic = "dict";
}

def PylirPy_FunctionAttr : AttrSpecBuilder<"Function",
	[ParamSpec<AttrParameter<"mlir::FlatSymbolRefAttr", "">, "value">,
	 ParamSpec<DefaultValuedParameter<"mlir::Attribute", "::pylir::Py::StrAttr::get($_ctx, \"\")">, "qualName">,
	 ParamSpec<DefaultValuedParameter<"mlir::Attribute", "::mlir::FlatSymbolRefAttr::get($_ctx, pylir::Py::Builtins::None.name)">, "defaults">,
	 ParamSpec<DefaultValuedParameter<"mlir::Attribute", "::mlir::FlatSymbolRefAttr::get($_ctx, pylir::Py::Builtins::None.name)">, "kwDefaults">,
	 ParamSpec<DefaultValuedParameter<"mlir::Attribute", "::mlir::Attribute{}">, "dict">
	 ], [DeclareAttrInterfaceMethods<SubElementAttrInterface, ["replaceImmediateSubAttribute"]>,
	     DeclareAttrInterfaceMethods<ObjectAttrInterface>]>
{
	let mnemonic = "function";
}

def PylirPy_TypeAttr : TypeObjectSlotsAttr<"Type",
	[ParamSpec<DefaultValuedParameter<"::mlir::Attribute", "::pylir::Py::TupleAttr::get($_ctx, {})">, "mroTuple">]> {
	let mnemonic = "type";
}

def PylirPy_UnboundAttr : PylirPy_Attr<"Unbound"> {
    let mnemonic = "unbound";
    let summary = "python unbound value";
}

#endif
