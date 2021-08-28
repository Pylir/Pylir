
#ifndef PYLIR_PY_ATTR_TABLEGEN
#define PYLIR_PY_ATTR_TABLEGEN


include "pylir/Optimizer/PylirPy/IR/PylirPyBase.td"

class PylirPy_Attr<string name, list<Trait> traits = [], string baseCppClass = "::mlir::Attribute"> :
    AttrDef<PylirPy_Dialect, name, traits, baseCppClass> {
    let storageType = dialect.cppNamespace # "::" # cppClassName;
}

def PylirPy_ListAttr : PylirPy_Attr<"List"> {
    let mnemonic = "list";

    let returnType = "llvm::ArrayRef<mlir::Attribute>";
    let convertFromStorage = "$_self.getValue()";
    let constBuilderCall = storageType # "::get($_builder.getContext(), $0)";

    let parameters = (ins ArrayRefParameter<"::mlir::Attribute">:$value);
}

def PylirPy_TupleAttr : PylirPy_Attr<"Tuple"> {
    let mnemonic = "tuple";

    let returnType = "llvm::ArrayRef<mlir::Attribute>";
    let convertFromStorage = "$_self.getValue()";
    let constBuilderCall = storageType # "::get($_builder.getContext(), $0)";

    let parameters = (ins ArrayRefParameter<"::mlir::Attribute">:$value);
}

def PylirPy_SetAttr : PylirPy_Attr<"Set"> {
    let mnemonic = "set";

    let returnType = "llvm::ArrayRef<mlir::Attribute>";
    let convertFromStorage = "$_self.getValue()";
    let constBuilderCall = storageType # "::get($_builder.getContext(), $0)";

    let parameters = (ins ArrayRefParameter<"::mlir::Attribute">:$value);
}

def PylirPy_DictAttr : PylirPy_Attr<"Dict"> {
    let mnemonic = "dict";

    let returnType = "llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>>";
    let convertFromStorage = "$_self.getValue()";
    let constBuilderCall = storageType # "::get($_builder.getContext(), $0)";

    let parameters = (ins ArrayRefParameter<"std::pair<mlir::Attribute, mlir::Attribute>">:$value);
}

class AnyAttrOf<list<Attr> allowedAttrs, string summary = "", string cppClassName = "::mlir::Attribute"> : Attr<
    Or<!foreach(allowedAttr,allowedAttrs,allowedAttr.predicate)>,
    !if(!eq(summary, ""),
            !interleave(!foreach(t, allowedAttrs, t.summary), " or "),
            summary)
> {
    let storageType = "::mlir::Attribute";
    let returnType = "::mlir::Attribute";
    let convertFromStorage = "$_self";
    let constBuilderCall = "$0";
}

def AnyPyAttribute : AnyAttrOf<[APIntAttr, BoolAttr, F64Attr, PylirPy_ListAttr,
                                    PylirPy_TupleAttr, PylirPy_SetAttr, PylirPy_DictAttr, StrAttr, FlatSymbolRefAttr]>;
#endif
