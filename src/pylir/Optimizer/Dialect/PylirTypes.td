
#ifndef PYLIR_TYPES_TABLEGEN
#define PYLIR_TYPES_TABLEGEN

include "pylir/Optimizer/Dialect/PylirBase.td"
include "mlir/IR/SubElementInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

class Pylir_Type<string name, list<Trait> traits = [], string baseCppClass = "::mlir::Type"> :
    TypeDef<Pylir_Dialect, name, traits, baseCppClass>;

def Pylir_Integer : Pylir_Type<"Integer"> {
    let mnemonic = "int";
    let summary = "python big integer";
}

def Pylir_String : Pylir_Type<"String"> {
    let mnemonic = "str";
    let summary = "python string";
}

def Pylir_List : Pylir_Type<"List", []> {
    let mnemonic = "list";
    let summary = "python list";
}

def Pylir_Set : Pylir_Type<"Set", []> {
    let mnemonic = "set";
    let summary = "python set";
}

def Pylir_Tuple : Pylir_Type<"Tuple"> {
    let mnemonic = "tuple";
    let summary = "python tuple";
}

def Pylir_Dict : Pylir_Type<"Dict", []> {
    let mnemonic = "dict";
    let summary = "python dictionary";
}

def Pylir_Object : Pylir_Type<"Object", [DeclareTypeInterfaceMethods<SubElementTypeInterface>]> {
    let mnemonic = "object";
    let parameters = (ins "::mlir::FlatSymbolRefAttr":$type);

    let parser = [{
        if ($_parser.parseOptionalLess())
        {
            return get($_ctxt);
        }
        mlir::FlatSymbolRefAttr attribute;
        if ($_parser.parseAttribute(attribute) || $_parser.parseGreater())
        {
            return {};
        }
        return get(attribute);
    }];

    let printer = [{
        $_printer << getMnemonic();
        if (!getType())
        {
            return;
        }
        $_printer << '<' << getType() << '>';
    }];

    let builders = [
            TypeBuilder<(ins ), [{
                  return Base::get($_ctxt, mlir::FlatSymbolRefAttr{});
                }]>,
            TypeBuilderWithInferredContext<(ins "::mlir::FlatSymbolRefAttr":$type), [{
              return Base::get(type.getContext(), type);
            }]>,
            TypeBuilderWithInferredContext<(ins "::mlir::SymbolOpInterface":$symbol), [{
              return get(mlir::FlatSymbolRefAttr::get(symbol.getContext(), symbol.getName()));
            }]>
          ];

    let skipDefaultBuilders = 1;

    let summary = "python object";
}

def Pylir_Pointer : Pylir_Type<"Pointer", [DeclareTypeInterfaceMethods<SubElementTypeInterface>]> {
    let mnemonic = "pointer";
    let parameters = (ins "::mlir::Type":$elementType);

    let parser = [{
            if ($_parser.parseLess())
            {
                return {};
            }
            mlir::Type type;
            if ($_parser.parseType(type) || $_parser.parseGreater())
            {
                return {};
            }
            return get(type);
    }];

    let printer = [{
            $_printer << getMnemonic() << '<' << getElementType() << '>';
     }];

    let builders = [
        TypeBuilderWithInferredContext<(ins "::mlir::Type":$type), [{
                  return Base::get(type.getContext(), type);
                }]>
    ];

    let skipDefaultBuilders = 1;
}

def Pylir_KnownTypeObject : TypeConstraint<
    And<[
        CPred<"$_self.isa<::pylir::Dialect::ObjectType>()">,
        CPred<"$_self.cast<::pylir::Dialect::ObjectType>().getType()">
    ]>,
    "python object with a statically known type",
    "::pylir::Dialect::ObjectType"
>;

def Pylir_KnownTypeObjectAttr : Attr<
    And<[
        TypeAttr.predicate,
        SubstLeaves<"$_self", "$_self.cast<mlir::TypeAttr>().getValue()", Pylir_KnownTypeObject.predicate>
    ]>
> {
    let storageType = TypeAttr.storageType;
    let returnType = "::pylir::Dialect::ObjectType";
    let valueType = NoneType;
    let constBuilderCall = "::mlir::TypeAttr::get($0)";
    let convertFromStorage = "$_self.getValue().cast<::pylir::Dialect::ObjectType>()";
}

class Pylir_KnownTypeObjectCheck<string typeObjectNameVariable> : Type<
    And<[
        Pylir_KnownTypeObject.predicate,
        CPred<"$_self.cast<::pylir::Dialect::ObjectType>().getType().getValue() =="
        "::llvm::StringRef{::pylir::Dialect::" # typeObjectNameVariable # "}">
        ]>, "python object with type object pylir::Dialect::" # typeObjectNameVariable, "::pylir::Dialect::ObjectType"
> {
    let builderCall = "::pylir::Dialect::ObjectType::get($_builder.getSymbolRefAttr(::pylir::Dialect::" # typeObjectNameVariable # "))";
}

def Pylir_TypeObject : Pylir_KnownTypeObjectCheck<"typeTypeObjectName">;

def Pylir_Bool : TypeAlias<I1>;

def Pylir_Float : TypeAlias<F64>;

class PointerOf<Type type> : Type<
    And<[
        Pylir_Pointer.predicate,
        SubstLeaves<"$_self", "$_self.cast<::pylir::Dialect::PointerType>().getElementType()", type.predicate>
    ]>, " pointer to " # type.summary,"::pylir::Dialect::PointerType"
> {
    let builderCall = !if(!empty(type.builderCall),"","::pylir::Dialect::PointerType::get(" # type.builderCall # ")");
}

def AnyPointer : PointerOf<AnyType>;

def NotUnknownObjectPointer : TypeConstraint<And<[
    PointerOf<AnyType>.predicate,
    Or<[
    Neg<Pylir_Object.predicate>,
    SubstLeaves<"$_self", "$_self.cast<pylir::Dialect::PointerType>().getElementType()", Pylir_KnownTypeObject.predicate>
    ]>
]>, "Pointer to known object type or other","pylir::Dialect::PointerType">;

def Pylir_Function : Type<
    And<[
        FunctionType.predicate,
        CPred<"$_self.cast<::mlir::FunctionType>().getNumResults() == 1">,
        SubstLeaves<"$_self", "$_self.cast<::mlir::FunctionType>().getResult(0)", PointerOf<Pylir_Object>.predicate>,
        CPred<"$_self.cast<::mlir::FunctionType>().getNumInputs() == 3">,
        SubstLeaves<"$_self", "$_self.cast<::mlir::FunctionType>().getInput(0)", PointerOf<Pylir_Object>.predicate>,
        SubstLeaves<"$_self", "$_self.cast<::mlir::FunctionType>().getInput(1)", Pylir_Tuple.predicate>,
        SubstLeaves<"$_self", "$_self.cast<::mlir::FunctionType>().getInput(2)", PointerOf<Pylir_Dict>.predicate>
    ]>>
{
    let summary = "python function";
    let builderCall = [{
        mlir::FunctionType::get(
            $_builder.getContext(),
            {::pylir::Dialect::get(ObjectType::get($_builder.getContext())),
             Dialect::TupleType::get($_builder.getContext()),
             ::pylir::Dialect::get(Dialect::DictType::get($_builder.getContext()))},
            {::pylir::Dialect::getget(ObjectType::get($_builder.getContext()))})
    }];
}

def AnySequenceType : AnyTypeOf<[Pylir_String, Pylir_Tuple, Pylir_List]>;

def AnyImmutableType : AnyTypeOf<[Pylir_Bool, Pylir_Float, Pylir_Integer, Pylir_String, Pylir_Tuple, Pylir_Function]>;

def AnyMutableType : AnyTypeOf<[Pylir_List, Pylir_Set, Pylir_Dict]>;

def AnyValueType : AnyTypeOf<[AnyImmutableType, AnyMutableType]>;

#endif
