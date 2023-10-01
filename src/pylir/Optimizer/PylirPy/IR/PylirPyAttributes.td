// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_PY_ATTR_TABLEGEN
#define PYLIR_PY_ATTR_TABLEGEN

include "pylir/Optimizer/PylirPy/IR/PylirPyAttrInterfaces.td"
include "pylir/Optimizer/PylirPy/IR/PylirPyBase.td"
include "pylir/Optimizer/PylirPy/IR/PylirPyTraits.td"
include "pylir/Optimizer/Interfaces/SROAInterfaces.td"
include "pylir/Optimizer/Interfaces/AttrVerifyInterface.td"
include "pylir/Optimizer/Util/TablegenUtil.td"
include "mlir/IR/OpBase.td"

class PylirPy_Attr<string name, list<Trait> traits = [],
    string baseCppClass = "::mlir::Attribute"> :
  AttrDef<PylirPy_Dialect, name, traits, baseCppClass>;

/// Dictionary parameter containing the slots of a python object attribute.
defvar SlotsMap = DefaultValuedParameter<"::mlir::DictionaryAttr",
  "::mlir::DictionaryAttr::get($_ctxt, {})">;

/// Trait used to automatically implement `ObjectAttrInterface` that returns
/// a `GlobalValueAttr` to `type`. This should only be used by builtins where the given
/// type object is guaranteed to exist.
class KnownTypeAttr<string type>
  : AttrInterfaceImplementation<ObjectAttrInterface, [{
  ::pylir::Py::ObjectBaseAttribute getTypeObject() const;
}], [{
  ::pylir::Py::ObjectBaseAttribute $cppClass::getTypeObject() const {
    return pylir::Py::GlobalValueAttr::get(getContext(), pylir::Builtins::}] # type # [{.name);
  }
}]>;

/// Trait used to automatically implement `ConstObjectAttrInterface` and return
/// an empty dictionary for slots.
def EmptySlotsAttr
  : AttrInterfaceImplementation<ConstObjectAttrInterface, [{
  ::mlir::DictionaryAttr getSlots() const {
    return mlir::DictionaryAttr::get(getContext());
  }
}]>;

/// Convenient base class for Python object attributes adding common methods and
/// parameters.
class PylirPy_PyObjAttr<string name,
  list<Trait> traits = [], string baseCppClass = "ConcreteObjectAttribute">
  : PylirPy_Attr<name, !listconcat([ConstObjectAttrInterface,
   KnownTypeAttr<name>, DeclareAttrInterfaceMethods<SROAAttrInterface>],
   traits), baseCppClass>;

def PylirPy_ObjectAttr : PylirPy_Attr<"Object", [ConstObjectAttrInterface,
  DeclareAttrInterfaceMethods<SROAAttrInterface>], "ConcreteObjectAttribute"> {
	let mnemonic = "obj";
	let summary = "python object";

	let parameters = (ins "ObjectBaseAttribute":$type_object, SlotsMap:$slots);

	let skipDefaultBuilders = 1;

	let builders = [
    AttrBuilder<(ins "ObjectBaseAttribute":$typeObject,
      CArg<"mlir::DictionaryAttr", "{}">:$slots), [{
      slots = slots ? slots : DictionaryAttr::get($_ctxt);
  	  return $_get($_ctxt, typeObject, slots);
    }]>
  ];

	let assemblyFormat = "`<` $type_object (`,` $slots^)? `>`";
}

def BigIntParam : AttrParameter<"::pylir::BigInt", "Arbitrary sized integer",
  "const pylir::BigInt&"> {
  let printer = "$_printer << $_self.toString();";
}

def PylirPy_IntAttr : PylirPy_PyObjAttr<"Int", [IntAttrInterface,
  EmptySlotsAttr, DeclareAttrInterfaceMethods<EqualsAttrInterface>]> {
	let mnemonic = "int";
	let summary = "python integer";

  let parameters = (ins BigIntParam:$value);

	let assemblyFormat = "`<` $value `>`";

	let constBuilderCall = "::pylir::Py::IntAttr::get($_builder.getContext(), $0)";

  let extraClassDeclaration = [{

    //===------------------------------------------------------------------===//
    // IntAttrInterface implementation
    //===------------------------------------------------------------------===//

    const BigInt& getInteger() const {
      return getValue();
    }
  }];

	let returnType = "const ::pylir::BigInt&";
  let convertFromStorage = "$_self.getValue()";
}

def BoolParameter : AttrParameter<"bool", "boolean"> {
  let printer = [{$_printer << ($_self ? "True" : "False")}];
  let parser = [{
    mlir::AsmParser::KeywordSwitch<mlir::FailureOr<bool>>($_parser)
      .Case("True", true)
      .Case("False", false)
      .Default(mlir::failure())
  }];
}

def PylirPy_BoolAttr : PylirPy_PyObjAttr<"Bool", [BoolAttrInterface,
  EmptySlotsAttr,  DeclareAttrInterfaceMethods<EqualsAttrInterface>]> {
	let mnemonic = "bool";
	let summary = "python boolean";

  let parameters = (ins BoolParameter:$value);

	let assemblyFormat = "`<` $value `>`";

	let constBuilderCall = [{
	  ::pylir::Py::BoolAttr::get($_builder.getContext(), $0)
	}];

  let extraClassDeclaration = [{

    //===------------------------------------------------------------------===//
    // IntAttrInterface implementation
    //===------------------------------------------------------------------===//

    BigInt getInteger() const {
      return BigInt(getValue());
    }

    //===------------------------------------------------------------------===//
    // BoolAttrInterface implementation
    //===------------------------------------------------------------------===//

    bool getBoolean() const {
      return getValue();
    }
  }];

	let returnType = "bool";
  let convertFromStorage = "$_self.getValue()";
}

def PylirPy_FloatAttr : PylirPy_PyObjAttr<"Float", [FloatAttrInterface,
  EmptySlotsAttr,  DeclareAttrInterfaceMethods<EqualsAttrInterface>]> {
	let mnemonic = "float";
	let summary = "python float";

  let parameters = (ins APFloatParameter<"">:$value);

  let extraClassDeclaration = [{
    double getDoubleValue() const {
      return getValue().convertToDouble();
    }
  }];

  let assemblyFormat = "`<` $value `>`";

  let returnType = "double";
  let convertFromStorage = "$_self.getDoubleValue()";
}

def PylirPy_StrAttr : PylirPy_PyObjAttr<"Str", [StrAttrInterface,
  EmptySlotsAttr,  DeclareAttrInterfaceMethods<EqualsAttrInterface>]> {
	let mnemonic = "str";
	let summary = "python string";

  let parameters = (ins StringRefParameter<>:$value);

	let assemblyFormat = "`<` $value `>`";
}

def PylirPy_TupleAttr : PylirPy_PyObjAttr<"Tuple", [TupleAttrInterface,
  EmptySlotsAttr]> {
	let mnemonic = "tuple";
	let summary = "python tuple";

  let parameters = (ins OptionalArrayRefParameter<"mlir::Attribute">:$elements);

	let assemblyFormat = [{
	  `<` `(` (`)`) : ($elements^ `)`)? `>`
	}];

  let skipDefaultBuilders = 1;

  let builders = [
    AttrBuilder<(ins CArg<"llvm::ArrayRef<mlir::Attribute>", "{}">:$elements),
    [{
      return $_get($_ctxt, elements);
    }]>
  ];
}

def PylirPy_ListAttr : PylirPy_PyObjAttr<"List", [EmptySlotsAttr]> {
	let mnemonic = "list";

  let parameters = (ins OptionalArrayRefParameter<"mlir::Attribute">:$elements);

	let assemblyFormat = [{
	  `<` `[` (`]`) : ( $elements^ `]`)? `>`
	}];

	let skipDefaultBuilders = 1;

	let builders = [
    AttrBuilder<(ins CArg<"llvm::ArrayRef<mlir::Attribute>", "{}">:$elements),
    [{
      return $_get($_ctxt, elements);
    }]>
  ];
}

def PylirPy_DictAttr : PylirPy_PyObjAttr<"Dict", [DictAttrInterface,
  EmptySlotsAttr, DeclareAttrInterfaceMethods<AttrVerifyInterface>]> {

  let mnemonic = "dict";
  let summary = "python dictionary";

  let description = [{
    Attribute for representing a constant dictionary.
    Like most `ObjectAttrInterface` implementations it is also capable of having
    a custom `typeObject` as well as `slots`.
    If a `typeObject` is specified, it has to be a subclass of `builtins.dict`.
    If unspecified, it defaults to `builtins.dict`.

    `DictAttr` semantically behaves the same as python dictionaries:
    It uniques the key-value pairs based on the key, ensuring only the latest
    entry is present, and retains insertion order of all key-value pairs while
    doing so.
    The deduplication already happens during parsing.

    The most important detail about using `DictAttr` is understanding the
    restrictions on the keys.
    Attributes that are passed in as keys need to have both a known `__hash__`
    implementation as well as `__eq__` implementation
    (as is determined by `getHashFunction` and `getEqualsFunction`).
    It is therefore not possible to have arbitrary keys in a `DictAttr`.

    Additionally, it is necessary to also specify a "normalized key" when
    constructing a `DictAttr` or within the IR syntax, if using a `GlobalValueAttr` as
    key that does not use identity equality as equals function.
    The normalized key is the result of calling `getCanonicalEqualsForm` on the
    actual key of an entry and is what is used to determine whether two keys are
    equal. Given a `GlobalValueAttr` referring to a `py.globalValue` with the given
    equals function of the initializer argument, the following attribute kinds
    are returned by `getCanonicalEqualsForm`:
    * `py.int`, `py.float` -> `py.frac`
    * `py.str` -> "string literal"

    Syntax:
    ```
    dict ::= `#py.dict` `<` `{` { attr [`norm` attr] `to` attr } `}` `>`
    ```

    Examples:
    ```
    #py.dict<{#py.ref<@intGlobalValue> norm #py.frac<3,5> to #py.str<"value">}>
    #py.dict<{#py.ref<@strGlobalValue> norm "the text" to #py.str<"value">}>
    #py.dict<{#py.int<5> to #py.str<"value">}>
    ```
  }];

  let parameters = (ins
    ArrayRefParameter<"std::pair<mlir::Attribute, std::size_t>">:
      $normalized_keys_internal,
    ArrayRefParameter<"std::pair<mlir::Attribute, mlir::Attribute>">:
      $key_value_pairs);

  let builders = [
    AttrBuilder<(ins CArg<"llvm::ArrayRef<Entry>", "{}">:$entries)>
  ];

  let assemblyFormat = [{
    `<` custom<KVPair>($key_value_pairs, $normalized_keys_internal) `>`
  }];

  let extraClassDeclaration = [{

    /// Tag used in 'Entry' to signal that the normalized key should be deduced.
    struct DeduceNormalizedTag{};

    struct Entry
    {
      mlir::Attribute key;
      mlir::Attribute normalizedKey;
      mlir::Attribute value;

      /// Constructs an entry by explicitly specifying the key, its normalized
      /// value and the mapped to value.
      Entry(mlir::Attribute key, mlir::Attribute normalizedKey,
          mlir::Attribute value) : key(key), normalizedKey(normalizedKey),
          value(value)
      {}

      /// Constructs an entry by specifying the key and the mapped to value.
      /// The normalized value will be deduced from the key.
      Entry(EqualsAttrInterface key, mlir::Attribute value)
        : key(key), normalizedKey(key.getCanonicalAttribute()), value(value)
      {}
    };

    /// Returns the value for a given key in the dictionary. Returns a null
    /// attribute if not found. The key must be a 'GlobalValueAttr' or
    /// 'ObjectAttrInterface'.
    mlir::Attribute lookup(mlir::Attribute key) const;

    /// Returns a range of all the normalized keys. The order is unspecified.
    auto getNormalizedKeys() const {
      return llvm::make_filter_range(llvm::map_range(getNormalizedKeysInternal(),
        +[](std::pair<mlir::Attribute, std::size_t> pair) {
          return pair.first;
        }), +[](mlir::Attribute attr) { return static_cast<bool>(attr); });
    }
  }];
}

def PylirPy_FunctionAttr : PylirPy_PyObjAttr<"Function",
  [FunctionAttrInterface, DeclareAttrInterfaceMethods<AttrVerifyInterface>]> {
	let mnemonic = "function";
	let summary = "python function";

	let parameters = (ins
	    "mlir::FlatSymbolRefAttr":$value,
	    DefaultValuedParameter<"mlir::Attribute",
	        "StrAttr::get($_ctxt, \"\")">:$qual_name,
	    DefaultValuedParameter<"mlir::Attribute",
	        "GlobalValueAttr::get($_ctxt, Builtins::None.name)">:$defaults,
	    DefaultValuedParameter<"mlir::Attribute",
	        "GlobalValueAttr::get($_ctxt, Builtins::None.name)">:$kw_defaults,
	    DefaultValuedParameter<"mlir::Attribute", "mlir::Attribute{}">:$dict
  );

	let assemblyFormat = [{
	  `<` $value ( `,` struct($qual_name, $defaults, $kw_defaults, $dict)^)? `>`
	}];

	let builders = [
    AttrBuilderWithInferredContext<(ins
            "mlir::FlatSymbolRefAttr":$value,
             CArg<"::mlir::Attribute", "{}">:$qualName,
             CArg<"::mlir::Attribute", "{}">:$defaults,
             CArg<"::mlir::Attribute", "{}">:$kwDefaults,
             CArg<"::mlir::Attribute", "{}">:$dict), [{
      mlir::MLIRContext* context = value.getContext();
      qualName = qualName ? qualName : StrAttr::get(context, "");
      defaults = defaults ? defaults
        : GlobalValueAttr::get(context, Builtins::None.name);
      kwDefaults = kwDefaults ? kwDefaults
        : GlobalValueAttr::get(context, Builtins::None.name);
      return $_get(context, value, qualName, defaults, kwDefaults, dict);
    }]>
  ];

  let extraClassDeclaration = [{
    mlir::DictionaryAttr getSlots() const;
  }];
}

def PylirPy_TypeAttr : PylirPy_PyObjAttr<"Type", [TypeAttrInterface,
  DeclareAttrInterfaceMethods<AttrVerifyInterface>]> {
	let mnemonic = "type";
	let summary = "python type";

  let parameters = (ins
    DefaultValuedParameter<"mlir::Attribute", "TupleAttr::get($_ctxt, {})">:
      $mro_tuple,
    DefaultValuedParameter<"TupleAttr", "TupleAttr::get($_ctxt, {})">:
      $instance_slots,
    SlotsMap:$slots
  );

  let skipDefaultBuilders = 1;

  let builders = [
    AttrBuilder<(ins
      CArg<"mlir::Attribute", "{}">:$mroTuple,
      CArg<"TupleAttr", "{}">:$instanceSlots,
      CArg<"mlir::DictionaryAttr", "{}">:$slots), [{
      mroTuple = mroTuple ? mroTuple : TupleAttr::get($_ctxt);
      instanceSlots = instanceSlots ? instanceSlots : TupleAttr::get($_ctxt);
      slots = slots ? slots : DictionaryAttr::get($_ctxt);
      return $_get($_ctxt, mroTuple, instanceSlots, slots);
    }]>
  ];

  let description = [{
    Attribute for representing Python `type` objects and subclasses.
    For the most part, it acts as any other object attribute, having both a
    customizable type object as well as slots.

    Additionally, it contains two parameters. `mroTuple` is a tuple of type
    objects (or references to type objects) forming the order in which method
    lookups are resolved.
    `instance_slots` is a tuple of strings, which denotes the slots an instance
    of this type has.
    If neither of the two parameters are specified (in the C++ builder or in
    IR syntax), they default to an empty tuple.

    Syntax:
    ```
    optional-attrs ::= (`slots` | `mro_tuple` | `instance_slots`) `=` attr
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

def PylirPy_FractionalAttr : PylirPy_Attr<"Fractional"> {

  let mnemonic = "frac";

  let description = [{

  }];

  let parameters = (ins BigIntParam:$nominator, BigIntParam:$denominator);

  let assemblyFormat = "`<` $nominator `,` $denominator `>`";
}

def PylirPy_GlobalValueAttr : PylirPy_Attr<"GlobalValue",
    [NativeAttrTrait<"IsMutable">], "ObjectBaseAttribute"> {
  let mnemonic = "globalValue";

  let description = [{
    A `#py.globalValue` is a special kind of attribute, representing an instance
    of a globally accessibly Python object.
    The op is optionally initialized with `$initializer`, dictating the type and
    initial value of the global value.
    If no `$initializer` is given, the op must have been marked external using a
    `py.external` in the module.
    It differs from other attributes in the following way:
    * It is mutable unless marked with `const`.
      The attribute can be turned into an SSA-value using `py.constant` and
      mutated with the usual operations.
      If marked `const`, any operations writing to the `py.globalValue` leads to
      undefined behaviour.
    * It has nominal identity instead of structural identity.
      This means that two `#py.globalValue` instances that have different names
      are considered different objects.
      A `py.is` operation comparing these would therefore result in `False`.

    Syntax:
    ```text
    global-value ::= `#py.globalValue` `<` name { (`,` `const`)? (`,` `initializer` `=` attr) } `>`
    ```

    Examples:
    ```text
    #py.globalValue<builtins.str, const, initializer = #py.type<mro_tuple = #py.tuple<()>>>
    #py.globalValue<builtins.list>
    #py.globalValue<builtins.type, initializer = #py.type>
    ```
  }];

  let parameters = (ins StringRefParameter<>:$name,
              DefaultValuedParameter<"bool", "false">:$constant,
              OptionalParameter<"ConcreteObjectAttribute">:$initializer);

  // Required to make it properly mutable.
  let genStorageClass = 0;

  // Don't want to use the default builder as it includes the mutable
  // parameters.
  let skipDefaultBuilders = 1;

  // Have to do a completely custom parse and print for the time being as for a
  // variety of reasons, mutable cyclic attributes cannot be parsed and printed
  // by the ODS generated prints and parsers at this point in time.
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Sets whether this global value is constant or not.
    void setConstant(bool constant);

    /// Sets the initializer of this global value.
    void setInitializer(ConcreteObjectAttribute initializer);
  }];

  let builders = [
    AttrBuilder<(ins "::llvm::StringRef":$name), [{
      return $_get($_ctxt, name);
    }]>,
  ];

  let constBuilderCall = [{
    ::pylir::Py::GlobalValueAttr::get($_builder.getContext(), $0)
  }];
}

#endif
