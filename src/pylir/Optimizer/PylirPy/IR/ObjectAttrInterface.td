#ifndef PYLIRPY_INTERFACES_OBJECT_ATTR_INTERFACE
#define PYLIRPY_INTERFACES_OBJECT_ATTR_INTERFACE

include "mlir/IR/OpBase.td"

def ObjectAttrInterface : AttrInterface<"ObjectAttrInterface"> {
    let cppNamespace = "::pylir::Py";

    let methods = [
        InterfaceMethod<[{}], "::mlir::FlatSymbolRefAttr", "getTypeObject", (ins)>,
        InterfaceMethod<[{}], "::mlir::DictionaryAttr", "getSlots", (ins)>
    ];
}

#endif
