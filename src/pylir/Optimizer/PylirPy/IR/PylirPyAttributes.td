
#ifndef PYLIR_PY_ATTR_TABLEGEN
#define PYLIR_PY_ATTR_TABLEGEN

include "pylir/Optimizer/PylirPy/IR/PylirPyBase.td"
include "mlir/IR/SubElementInterfaces.td"

class PylirPy_Attr<string name, list<Trait> traits = [], string baseCppClass = "::mlir::Attribute"> :
    AttrDef<PylirPy_Dialect, name, traits, baseCppClass>;

def PylirPy_IntImplAttr : PylirPy_Attr<"IntImpl"> {
    let parameters = (ins "::pylir::BigInt":$value);

    let genStorageClass = 0;
}

def PylirPy_DictImplAttr : PylirPy_Attr<"DictImpl", [DeclareAttrInterfaceMethods<SubElementAttrInterface,
                                                                            ["replaceImmediateSubAttribute"]>]> {
    let parameters = (ins ArrayRefParameter<"std::pair<mlir::Attribute, mlir::Attribute>">:$value);

    let skipDefaultBuilders = 1;

    let extraClassDeclaration = [{
        static ::pylir::Py::DictImplAttr get(::mlir::MLIRContext* context, llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> attributes = {});

        static ::pylir::Py::DictImplAttr getUniqued(::mlir::MLIRContext* context, llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> attributes = {});
    }];
}

def PylirPy_ObjectAttr : PylirPy_Attr<"Object", [DeclareAttrInterfaceMethods<SubElementAttrInterface,
                                                                                ["replaceImmediateSubAttribute"]>]> {
    let mnemonic = "obj";
    let summary = "python object";

    let parameters = (ins "mlir::FlatSymbolRefAttr":$type, "::llvm::Optional<::pylir::Py::DictAttr>":$attributes, "::llvm::Optional<mlir::Attribute>":$builtinValue);
}

def PylirPy_UnboundAttr : PylirPy_Attr<"Unbound"> {
    let mnemonic = "unbound";
    let summary = "python unbound value";
}

def AnyPyAttribute : AnyAttrOf<[PylirPy_ObjectAttr, PylirPy_UnboundAttr, FlatSymbolRefAttr]>;

#endif
