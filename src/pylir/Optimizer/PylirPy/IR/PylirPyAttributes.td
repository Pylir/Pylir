// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_PY_ATTR_TABLEGEN
#define PYLIR_PY_ATTR_TABLEGEN

include "pylir/Optimizer/PylirPy/IR/PylirPyBase.td"
include "pylir/Optimizer/PylirPy/IR/ObjectAttrInterface.td"
include "pylir/Optimizer/PylirPy/IR/PylirPyTraits.td"
include "pylir/Optimizer/Interfaces/SROAInterfaces.td"
include "pylir/Optimizer/Util/TablegenUtil.td"
include "mlir/IR/SubElementInterfaces.td"
include "mlir/IR/OpBase.td"

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
										["replaceImmediateSubAttribute"]>, ObjectAttrInterface,
						 DeclareAttrInterfaceMethods<SROAAttrInterface>]),
										extraBuilderCode>;


class TypeObjectNoSlotsAttr<string name, list<ParamSpec> parameters, list<Trait> traits = []>
    : AttrSpecBuilder<name,
    !listconcat(parameters, [ParamSpec<DefaultType<name>, "typeObject">]),
    !listconcat(traits, [DeclareAttrInterfaceMethods<SubElementAttrInterface,
										["replaceImmediateSubAttribute"]>, ObjectAttrInterface,
						 DeclareAttrInterfaceMethods<SROAAttrInterface>])>;


def PylirPy_ObjectAttr : TypeObjectSlotsAttr<"Object", [], [
	DeclareAttrInterfaceMethods<SubElementAttrInterface, ["replaceImmediateSubAttribute"]>]> {
	let mnemonic = "obj";
	let summary = "python object";

	let assemblyFormat = "`<` $typeObject (`,` $slots^)? `>`";
}

def PylirPy_IntAttr : TypeObjectSlotsAttr<"Int", [
	ParamSpec<CustomPrint<AttrParameter<"pylir::BigInt", "", "const pylir::BigInt&">,
											?, "$_printer << $_self.toString();">, "value">],
	[ImmutableAttr]> {
	let mnemonic = "int";
	let summary = "python integer";

	let assemblyFormat = "`<` $value ( `,` struct($typeObject, $slots)^)? `>`";

	let constBuilderCall = "::pylir::Py::IntAttr::get($_builder.getContext(), $0)";

	let returnType = "const ::pylir::BigInt&";
    let convertFromStorage = "$_self.getValue()";
}

def PylirPy_BoolAttr : DialectAttr<PylirPy_Dialect, CPred<"$_self.isa<::pylir::Py::BoolAttr>()">> {
	let summary = "python boolean";
}

def PylirPy_FloatAttr : TypeObjectSlotsAttr<"Float", [ParamSpec<APFloatParameter<"">, "value">], [ImmutableAttr]> {
	let mnemonic = "float";
	let summary = "python float";

    let extraClassDeclaration = [{
        double getDoubleValue() const
        {
            return getValue().convertToDouble();
        }
    }];

    let assemblyFormat = "`<` $value ( `,` struct($typeObject, $slots)^)? `>`";

    let returnType = "double";
    let convertFromStorage = "$_self.getDoubleValue()";
}

def PylirPy_StrAttr : TypeObjectSlotsAttr<"Str", [ParamSpec<CustomPrint<StringRefParameter<>, ?, [{
	$_printer << "\"";
	llvm::printEscapedString($_self, $_printer.getStream());
	$_printer << "\"";
}]>, "value">], [ImmutableAttr]> {
	let mnemonic = "str";
	let summary = "python string";

	let assemblyFormat = "`<` $value ( `,` struct($typeObject, $slots)^)? `>`";
}

def PylirPy_TupleAttr : TypeObjectNoSlotsAttr<"Tuple",
	[ParamSpec<ListPrint<"::mlir::Attribute", "Paren">, "value", "{}">], [ImmutableAttr]> {
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

def PylirPy_DictAttr : PylirPy_Attr<"Dict", [DeclareAttrInterfaceMethods<SubElementAttrInterface,
    ["replaceImmediateSubAttribute"]>, ObjectAttrInterface, DeclareAttrInterfaceMethods<SROAAttrInterface>]> {

    let mnemonic = "dict";
    let summary = "python dictionary";

    let description = [{
        Attribute for representing a constant dictionary. Like most `ObjectAttrInterface` implementations it is also
        capable of having a custom `typeObject` as well as `slots`. If a `typeObject` is specified, it has to be a
        subclass of `builtins.dict`. If unspecified, it defaults to `builtins.dict`.

        `DictAttr` semantically behaves the same as python dictionaries: It uniques the key-value pairs based on the
        key, ensuring only the latest entry is present, and retains insertion order of all key-value pairs while doing
        so. The deduplication already happens during parsing.

        The most important detail about using `DictAttr` is understanding the restrictions on the keys. Attributes
        that are passed in as keys need to have both a known `__hash__` implementation as well as `__eq__`
        implementation (as is determined by `getHashFunction` and `getEqualsFunction`). It is therefore not possible to
        have arbitrary keys in a `DictAttr`.

        Additionally, it is necessary to also specify a "normalized key" when constructing a `DictAttr` or within
        the IR syntax, if using a `RefAttr` as key that does not use identity equality as equals function. The
        normalized key is the result of calling `getCanonicalEqualsForm` on the actual key of an entry and is what is
        used to determine whether two keys are equal.
        Given a `RefAttr` referring to a `py.globalValue` with the given equals function of the initializer argument,
        the following attribute kinds are returned by `getCanonicalEqualsForm`:
        * `py.int`, `py.float` -> `py.frac`
        * `py.str` -> "string literal"

        Syntax:
        ```
        optional-attrs ::= (`typeObject` | `slots`) `=` attr
        dict ::= `#py.dict` `<` `{` { attr [`norm` attr] `to` attr } `}` [`,` optional-attrs { `,` optional-attrs} ] `>`
        ```

        Examples:
        ```
        #py.dict<{#py.ref<@intGlobalValue> norm #py.frac<3,5> to #py.str<"value">}>
        #py.dict<{#py.ref<@strGlobalValue> norm "the text" to #py.str<"value">}>
        #py.dict<{#py.int<5> to #py.str<"value">}>
        ```
    }];

    let parameters = (ins ArrayRefParameter<"std::pair<mlir::Attribute, std::size_t>">:$normalized_keys_internal,
                          ArrayRefParameter<"std::pair<mlir::Attribute, mlir::Attribute>">:$key_value_pairs,
                          DefaultType<"Dict">:$typeObject, SlotsMap:$slots);

    let builders = [
        AttrBuilder<(ins CArg<"llvm::ArrayRef<Entry>", "{}">:$entries, CArg<"::pylir::Py::RefAttr", "{}">:$typeObject,
                         CArg<"mlir::DictionaryAttr", "{}">:$slots)>
    ];

    let assemblyFormat = [{
        `<` custom<KVPair>($key_value_pairs, $normalized_keys_internal) (`,` struct($typeObject, $slots)^)? `>`
    }];

    let extraClassDeclaration = [{

        /// Tag used in 'Entry' to signal that the normalized key should be deduced.
        struct DeduceNormalizedTag{};

        struct Entry
        {
            mlir::Attribute key;
            mlir::Attribute normalizedKey;
            mlir::Attribute value;

            /// Constructs an entry by explicitly specifying the key, its normalized value and the mapped to value.
            Entry(mlir::Attribute key, mlir::Attribute normalizedKey, mlir::Attribute value) : key(key),
                normalizedKey(normalizedKey), value(value)
            {}

            /// Constructs an entry by specifying the key and the mapped to value. The normalized value will be
            /// deduced from the key.
            Entry(DeduceNormalizedTag, mlir::Attribute key, mlir::Attribute value) : key(key), normalizedKey(nullptr),
                value(value)
            {}
        };

        /// Returns the value for a given key in the dictionary. Returns a null attribute if not found. The key must be
        /// a 'RefAttr' or 'ObjectAttrInterface'.
        mlir::Attribute lookup(mlir::Attribute key) const;

        /// Returns a range of all the normalized keys. The order is unspecified.
        auto getNormalizedKeys() const
        {
            return llvm::make_filter_range(llvm::map_range(getNormalizedKeysInternal(),
                +[](std::pair<mlir::Attribute, std::size_t> pair) {
                    return pair.first;
                }), +[](mlir::Attribute attr) { return static_cast<bool>(attr); });
        }
    }];
}

def PylirPy_FunctionAttr : AttrSpecBuilder<"Function",
	[ParamSpec<AttrParameter<"mlir::FlatSymbolRefAttr", "">, "value">,
	 ParamSpec<DefaultValuedParameter<"mlir::Attribute", "::pylir::Py::StrAttr::get($_ctxt, \"\")">, "qualName">,
	 ParamSpec<DefaultValuedParameter<"mlir::Attribute", "::pylir::Py::RefAttr::get($_ctxt, pylir::Builtins::None.name)">, "defaults">,
	 ParamSpec<DefaultValuedParameter<"mlir::Attribute", "::pylir::Py::RefAttr::get($_ctxt, pylir::Builtins::None.name)">, "kwDefaults">,
	 ParamSpec<DefaultValuedParameter<"mlir::Attribute", "::mlir::Attribute{}">, "dict">
	 ], [DeclareAttrInterfaceMethods<SubElementAttrInterface, ["replaceImmediateSubAttribute"]>,
	     DeclareAttrInterfaceMethods<ObjectAttrInterface>, DeclareAttrInterfaceMethods<SROAAttrInterface>,
	     ImmutableAttr]>
{
	let mnemonic = "function";
	let summary = "python function";

	let assemblyFormat = "`<` $value ( `,` struct($qualName, $defaults, $kwDefaults, $dict)^)? `>`";
}

def PylirPy_TypeAttr : TypeObjectSlotsAttr<"Type",
	[ParamSpec<DefaultValuedParameter<"::mlir::Attribute", "::pylir::Py::TupleAttr::get($_ctxt, {})">, "mroTuple">,
	 ParamSpec<DefaultValuedParameter<"::pylir::Py::TupleAttr", "::pylir::Py::TupleAttr::get($_ctxt, {})">,
	                                                                                                "instance_slots">],
	[ImmutableAttr]> {
	let mnemonic = "type";
	let summary = "python type";

    let description = [{
        Attribute for representing Python `type` objects and subclasses. For the most part, it acts as any other
        object attribute, having both a customizable type object as well as slots.

        Additionally, it contains two parameters. `mroTuple` is a tuple of type objects (or references to type objects)
        forming the order in which method lookups are resolved. `instance_slots` is a tuple of strings, which denotes
        the slots an instance of this type has.
        If neither of the two parameters are specified (in the C++ builder or in IR syntax), they default to an empty
        tuple.

        Syntax:
        ```
        optional-attrs ::= (`typeObject` | `slots` | `mroTuple` | `instance_slots`) `=` attr
        dict ::= `#py.type` [ `<` optional-attrs { `,` optional-attrs} ] `>` ]
        ```

        Examples:
        ```
        #py.type
        #py.type<instance_slots = #py.tuple<(#py.str<"lol">)>>
        #py.type<mroTuple = #py.tuple<(#py.ref<@lol>)>>
        ```
    }];

	let assemblyFormat = "(`<`struct(params)^`>`)?";
}

def PylirPy_UnboundAttr : PylirPy_Attr<"Unbound"> {
    let mnemonic = "unbound";
    let summary = "python unbound value";
}

def BigIntParam : AttrParameter<"::pylir::BigInt", "Arbitrary sized integer", "const pylir::BigInt&"> {
    let printer = "$_printer << $_self.toString();";
}

def PylirPy_FractionalAttr : PylirPy_Attr<"Fractional"> {

    let mnemonic = "frac";

    let description = [{

    }];

    let parameters = (ins BigIntParam:$nominator, BigIntParam:$denominator);

    let assemblyFormat = "`<` $nominator `,` $denominator `>`";
}

#endif
