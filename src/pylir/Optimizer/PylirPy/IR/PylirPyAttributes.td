
#ifndef PYLIR_PY_ATTR_TABLEGEN
#define PYLIR_PY_ATTR_TABLEGEN


include "pylir/Optimizer/PylirPy/IR/PylirPyBase.td"

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

def PylirPy_Singleton_None : I32EnumAttrCase<"None", 0>;
def PylirPy_Singleton_NotImplemented : I32EnumAttrCase<"NotImplemented", 1>;
def PylirPy_Singleton_Type : I32EnumAttrCase<"Type", 2, "type">;
def PylirPy_Singleton_Object : I32EnumAttrCase<"Object", 3, "object">;
def PylirPy_Singleton_Int : I32EnumAttrCase<"Int", 4, "int">;
def PylirPy_Singleton_Float : I32EnumAttrCase<"Float", 5, "float">;
def PylirPy_Singleton_Tuple : I32EnumAttrCase<"Tuple", 6, "tuple">;
def PylirPy_Singleton_List : I32EnumAttrCase<"List", 7, "list">;
def PylirPy_Singleton_Dict : I32EnumAttrCase<"Dict", 8, "dict">;
def PylirPy_Singleton_Cell : I32EnumAttrCase<"Cell", 9, "cell">;
def PylirPy_Singleton_NameError : I32EnumAttrCase<"NameError", 10>;
def PylirPy_Singleton_TypeError : I32EnumAttrCase<"TypeError", 11>;
def PylirPy_Singleton_UnboundLocalError : I32EnumAttrCase<"UnboundLocalError", 12>;
def PylirPy_Singleton_Function : I32EnumAttrCase<"Function", 13, "function">;

def PylirPy_SingletonAttr : I32EnumAttr<"SingletonKind", "", [
    PylirPy_Singleton_None, PylirPy_Singleton_NotImplemented, PylirPy_Singleton_Type, PylirPy_Singleton_Object,
    PylirPy_Singleton_Int, PylirPy_Singleton_Float, PylirPy_Singleton_Tuple, PylirPy_Singleton_List,
    PylirPy_Singleton_Dict, PylirPy_Singleton_Cell, PylirPy_Singleton_NameError, PylirPy_Singleton_TypeError,
    PylirPy_Singleton_UnboundLocalError, PylirPy_Singleton_Function
]> {
    let cppNamespace = "::pylir::Py";
}

def PylirPy_ObjectAttrPreDecl : Attr<CPred<"$_self.isa<::pylir::Py::ObjectAttr>()">> {
    let summary = "python object";

    let storageType = "::pylir::Py::ObjectAttr";
}

def AnyPyAttribute : AnyAttrOf<[PylirPy_IntAttr, PylirPy_BoolAttr, F64Attr, PylirPy_ListAttr, PylirPy_TupleAttr,
                                PylirPy_SetAttr, PylirPy_DictAttr, StrAttr, PylirPy_ObjectAttrPreDecl]>;

def PylirPy_ObjectAttr : PylirPy_Attr<"Object"> {
    let mnemonic = "dict";
    let summary = PylirPy_ObjectAttrPreDecl.summary;

    let parameters = (ins "mlir::Attribute":$type, "::pylir::Py::DictAttr":$attributes, "::llvm::Optional<mlir::Attribute>":$builtinValue);
}

#endif
