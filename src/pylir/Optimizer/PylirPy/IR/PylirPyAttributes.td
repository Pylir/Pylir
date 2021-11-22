
#ifndef PYLIR_PY_ATTR_TABLEGEN
#define PYLIR_PY_ATTR_TABLEGEN

include "pylir/Optimizer/PylirPy/IR/PylirPyBase.td"
include "mlir/IR/SubElementInterfaces.td"

class PylirPy_Attr<string name, list<Trait> traits = [], string baseCppClass = "::mlir::Attribute"> :
    AttrDef<PylirPy_Dialect, name, traits, baseCppClass>;

def PylirPy_IntAttr : PylirPy_Attr<"Int"> {
    let mnemonic = "int";
    let summary = "python integer";

    let returnType = "const ::pylir::BigInt&";
    let convertFromStorage = "$_self.getValue()";
    let constBuilderCall = storageType # "::get($_builder.getContext(), $0)";

    let parameters = (ins "::pylir::BigInt":$value);

    let genStorageClass = 0;

}

def PylirPy_ListAttr : PylirPy_Attr<"List", [DeclareAttrInterfaceMethods<SubElementAttrInterface,
                                                                            ["replaceImmediateSubAttribute"]>]> {
    let mnemonic = "list";
    let summary = "python list";

    let returnType = "llvm::ArrayRef<mlir::Attribute>";
    let convertFromStorage = "$_self.getValue()";
    let constBuilderCall = storageType # "::get($_builder.getContext(), $0)";

    let builders = [AttrBuilder<(ins ), "return get($_ctxt, {});">];

    let parameters = (ins ArrayRefParameter<"::mlir::Attribute">:$value);
}

def PylirPy_TupleAttr : PylirPy_Attr<"Tuple", [DeclareAttrInterfaceMethods<SubElementAttrInterface,
                                                                            ["replaceImmediateSubAttribute"]>]> {
    let mnemonic = "tuple";
    let summary = "python tuple";

    let returnType = "llvm::ArrayRef<mlir::Attribute>";
    let convertFromStorage = "$_self.getValue()";
    let constBuilderCall = storageType # "::get($_builder.getContext(), $0)";

    let builders = [AttrBuilder<(ins ), "return get($_ctxt, {});">];

    let parameters = (ins ArrayRefParameter<"::mlir::Attribute">:$value);
}

def PylirPy_SetAttr : PylirPy_Attr<"Set", [DeclareAttrInterfaceMethods<SubElementAttrInterface,
                                                                        ["replaceImmediateSubAttribute"]>]> {
    let mnemonic = "set";
    let summary = "python set";

    let returnType = "llvm::ArrayRef<mlir::Attribute>";
    let convertFromStorage = "$_self.getValue()";
    let constBuilderCall = storageType # "::get($_builder.getContext(), $0)";

    let parameters = (ins ArrayRefParameter<"::mlir::Attribute">:$value);

    let skipDefaultBuilders = 1;

    let extraClassDeclaration = [{
        static ::pylir::Py::SetAttr get(::mlir::MLIRContext* context, llvm::ArrayRef<mlir::Attribute> attributes = {});

        static ::pylir::Py::SetAttr getUniqued(::mlir::MLIRContext* context, llvm::ArrayRef<mlir::Attribute> attributes = {});
    }];
}

def PylirPy_DictAttr : PylirPy_Attr<"Dict", [DeclareAttrInterfaceMethods<SubElementAttrInterface,
                                                                            ["replaceImmediateSubAttribute"]>]> {
    let mnemonic = "dict";
    let summary = "python dictionary";

    let returnType = "llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>>";
    let convertFromStorage = "$_self.getValue()";
    let constBuilderCall = storageType # "::get($_builder.getContext(), $0)";

    let parameters = (ins ArrayRefParameter<"std::pair<mlir::Attribute, mlir::Attribute>">:$value);

    let skipDefaultBuilders = 1;

    let extraClassDeclaration = [{
        static ::pylir::Py::DictAttr get(::mlir::MLIRContext* context, llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> attributes = {});

        static ::pylir::Py::DictAttr getUniqued(::mlir::MLIRContext* context, llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> attributes = {});
    }];
}

def PylirPy_BoolAttr : Attr<CPred<"$_self.isa<::pylir::Py::BoolAttr>()">> {
    let summary = "python bool";

    let storageType = "::pylir::Py::BoolAttr";
    let returnType = "bool";
    let convertFromStorage = "$_self.getValue()";
    let constBuilderCall = "::pylir::Py::BoolAttr::get($_builder.getContext(),$0)";
}

def PylirPy_ObjectAttr : PylirPy_Attr<"Object", [DeclareAttrInterfaceMethods<SubElementAttrInterface,
                                                                                ["replaceImmediateSubAttribute"]>]> {
    let mnemonic = "obj";
    let summary = "python object";

    let parameters = (ins "mlir::Attribute":$type, "::pylir::Py::DictAttr":$attributes, "::llvm::Optional<mlir::Attribute>":$builtinValue);
}

def PylirPy_UnboundAttr : PylirPy_Attr<"Unbound"> {
    let mnemonic = "unbound";
    let summary = "python unbound value";
}

def AnyPyAttribute : AnyAttrOf<[PylirPy_IntAttr, PylirPy_BoolAttr, F64Attr, PylirPy_ListAttr, PylirPy_TupleAttr,
                                PylirPy_SetAttr, PylirPy_DictAttr, StrAttr, PylirPy_ObjectAttr, PylirPy_UnboundAttr,
                                FlatSymbolRefAttr]>;

#endif
