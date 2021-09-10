
#ifndef PYLIR_PY_ATTR_TABLEGEN
#define PYLIR_PY_ATTR_TABLEGEN


include "pylir/Optimizer/PylirPy/IR/PylirPyBase.td"

class PylirPy_Attr<string name, list<Trait> traits = [], string baseCppClass = "::mlir::Attribute"> :
    AttrDef<PylirPy_Dialect, name, traits, baseCppClass> {
    let storageType = dialect.cppNamespace # "::" # cppClassName;
}

def PylirPy_IntAttr : PylirPy_Attr<"Int"> {
    let mnemonic = "int";
    let summary = "python integer";

    let returnType = "const ::pylir::BigInt&";
    let convertFromStorage = "$_self.getValue()";
    let constBuilderCall = storageType # "::get($_builder.getContext(), $0)";

    let parameters = (ins "::pylir::BigInt":$value);

    let genStorageClass = 0;

}

def PylirPy_ListAttr : PylirPy_Attr<"List"> {
    let mnemonic = "list";
    let summary = "python list";

    let returnType = "llvm::ArrayRef<mlir::Attribute>";
    let convertFromStorage = "$_self.getValue()";
    let constBuilderCall = storageType # "::get($_builder.getContext(), $0)";

    let parameters = (ins ArrayRefParameter<"::mlir::Attribute">:$value);
}

def PylirPy_TupleAttr : PylirPy_Attr<"Tuple"> {
    let mnemonic = "tuple";
    let summary = "python tuple";

    let returnType = "llvm::ArrayRef<mlir::Attribute>";
    let convertFromStorage = "$_self.getValue()";
    let constBuilderCall = storageType # "::get($_builder.getContext(), $0)";

    let parameters = (ins ArrayRefParameter<"::mlir::Attribute">:$value);
}

def PylirPy_SetAttr : PylirPy_Attr<"Set"> {
    let mnemonic = "set";
    let summary = "python set";

    let returnType = "llvm::ArrayRef<mlir::Attribute>";
    let convertFromStorage = "$_self.getValue()";
    let constBuilderCall = storageType # "::get($_builder.getContext(), $0)";

    let parameters = (ins ArrayRefParameter<"::mlir::Attribute">:$value);
}

def PylirPy_DictAttr : PylirPy_Attr<"Dict"> {
    let mnemonic = "dict";
    let summary = "python dictionary";

    let returnType = "llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>>";
    let convertFromStorage = "$_self.getValue()";
    let constBuilderCall = storageType # "::get($_builder.getContext(), $0)";

    let parameters = (ins ArrayRefParameter<"std::pair<mlir::Attribute, mlir::Attribute>">:$value);
}

def PylirPy_BoolAttr : Attr<CPred<"$_self.isa<::pylir::Py::BoolAttr>()">> {
    let summary = "python bool";

    let storageType = "::pylir::Py::BoolAttr";
    let returnType = "bool";
    let convertFromStorage = "$_self.getValue()";
    let constBuilderCall = "::pylir::Py::BoolAttr::get($_builder.getContext(),$0)";
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

def AnyPyAttribute : AnyAttrOf<[PylirPy_IntAttr, PylirPy_BoolAttr, F64Attr, PylirPy_ListAttr,
                                    PylirPy_TupleAttr, PylirPy_SetAttr, PylirPy_DictAttr, StrAttr]>;
#endif
