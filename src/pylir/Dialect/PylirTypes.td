
#ifndef PYLIR_TYPES_TABLEGEN
#define PYLIR_TYPES_TABLEGEN

include "PylirBase.td"

class Pylir_Type<string name, string baseCppClass = "::mlir::Type"> :
    TypeDef<Pylir_Dialect, name, baseCppClass>;

def Pylir_Unknown : Pylir_Type<"Unknown"> {
    let mnemonic = "unknown";
}

def Pylir_Bool : Pylir_Type<"Bool"> {
    let mnemonic = "bool";
}

def Pylir_Float : Pylir_Type<"Float"> {
    let mnemonic = "float";
}

def Pylir_Integer : Pylir_Type<"Integer"> {
    let mnemonic = "int";
}

def Pylir_None : Pylir_Type<"None"> {
    let mnemonic = "none";
}

def Pylir_NotImplemented : Pylir_Type<"NotImplemented"> {
    let mnemonic = "notImplemented";
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

def Pylir_Variant : Pylir_Type<"Variant"> {
    let mnemonic = "variant";
    let parameters = (ins ArrayRefParameter<"::mlir::Type", "alternatives">
                          : $types);
    let parser = ?;
    let printer = ?;
    let genVerifyInvariantsDecl = 1;

    let builders = [
        TypeBuilderWithInferredContext<(ins "llvm::ArrayRef<mlir::Type>":$elementTypes), [{
          return Base::get(elementTypes[0].getContext(), detail::variantUnion(elementTypes));
        }], [{
          return Base::getChecked($_loc,  detail::variantUnion(elementTypes));
        }]>
      ];

    let skipDefaultBuilders = 1;
}

def Pylir_SlotObject : Pylir_Type<"SlotObject"> {
    let mnemonic = "slotobject";
    let parameters = (ins ArrayRefParameter<"std::pair<mlir::Attribute,mlir::Attribute>","slots">:$slots);

    let parser = ?;
    let printer = ?;
}

def AnySequenceType
: AnyTypeOf<[Pylir_String, Pylir_Tuple, Pylir_List]>;

def AnyPythonType
: AnyTypeOf<[Pylir_Unknown, Pylir_Bool, Pylir_Float, Pylir_Integer, Pylir_None,
             Pylir_NotImplemented, Pylir_String, Pylir_List, Pylir_Set, Pylir_Tuple,
             Pylir_Dict, Pylir_Variant]>;


def Pylir_Handle : Pylir_Type<"Handle"> {
    let mnemonic = "handle";
}

def Pylir_DictAttr : Attr<CPred<"$_self.isa<::pylir::Dialect::DictAttr>()">,"dict attribute"> {
    let storageType = [{ ::pylir::Dialect::DictAttr }];
    let returnType = [{ ::llvm::ArrayRef<std::pair<mlir::Attribute,mlir::Attribute>> }];
    let valueType = Pylir_Dict;
}

#endif
