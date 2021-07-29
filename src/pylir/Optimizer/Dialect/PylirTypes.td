
#ifndef PYLIR_TYPES_TABLEGEN
#define PYLIR_TYPES_TABLEGEN

include "pylir/Optimizer/Dialect/PylirBase.td"

class Pylir_Type<string name, list<Trait> traits = [], string baseCppClass = "::mlir::Type"> :
    TypeDef<Pylir_Dialect, name, traits, baseCppClass>;

def Pylir_Integer : Pylir_Type<"Integer"> {
    let mnemonic = "int";
}

def Pylir_String : Pylir_Type<"String"> {
    let mnemonic = "str";
}

def Pylir_List : Pylir_Type<"List"> {
    let mnemonic = "list";
}

def Pylir_Set : Pylir_Type<"Set"> {
    let mnemonic = "set";
}

def Pylir_Tuple : Pylir_Type<"Tuple"> {
    let mnemonic = "tuple";
}

def Pylir_Dict : Pylir_Type<"Dict"> {
    let mnemonic = "dict";
}

def Pylir_Function : Pylir_Type<"Function"> {
    let mnemonic = "function";
}

def Pylir_Object : Pylir_Type<"Object"> {
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
                  return Base::get($_ctxt);
                }]>,
            TypeBuilderWithInferredContext<(ins "::mlir::FlatSymbolRefAttr":$type), [{
              auto ret = Base::get(type.getContext());
              ret.setKnownType(type);
              return ret;
            }]>
          ];

    let skipDefaultBuilders = 1;
    let genStorageClass = 0;

    let extraClassDeclaration = [{
        void setKnownType(mlir::FlatSymbolRefAttr type);

        void clearType();
    }];
}

def Pylir_KnownTypeObject : TypeConstraint<
    And<[
        CPred<"$_self.isa<::pylir::Dialect::ObjectType>()">,
        CPred<"$_self.cast<::pylir::Dialect::ObjectType>().getType()">
    ]>,
    "",
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
        ]>,"","::pylir::Dialect::ObjectType"
> {
    let builderCall = "::pylir::Dialect::ObjectType::get($_builder.getSymbolRefAttr(::pylir::Dialect::" # typeObjectNameVariable # "))";
}

def Pylir_TypeObject : Pylir_KnownTypeObjectCheck<"typeTypeObjectName">;

def Pylir_Bool : TypeAlias<I1>;

def Pylir_Float : TypeAlias<F64>;

def AnySequenceType : AnyTypeOf<[Pylir_String, Pylir_Tuple, Pylir_List]>;

def AnyValueType : AnyTypeOf<[Pylir_Bool, Pylir_Float, Pylir_Integer, Pylir_String, Pylir_List, Pylir_Set, Pylir_Tuple,
                              Pylir_Dict, Pylir_Function]>;

def Pylir_Handle : Pylir_Type<"Handle"> {
    let mnemonic = "handle";
}

#endif
