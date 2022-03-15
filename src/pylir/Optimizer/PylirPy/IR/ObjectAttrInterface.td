#ifndef PYLIRPY_INTERFACES_OBJECT_ATTR_INTERFACE
#define PYLIRPY_INTERFACES_OBJECT_ATTR_INTERFACE

include "mlir/IR/OpBase.td"
include "pylir/Optimizer/Util/TablegenUtil.td"

def ObjectAttrInterface : UseableAttrInterface<"ObjectAttrInterface"> {
    let cppNamespace = "::pylir::Py";

    let methods = [
        InterfaceMethod<[{}], "::mlir::FlatSymbolRefAttr", "getTypeObject", (ins)>,
        InterfaceMethod<[{}], "::mlir::DictionaryAttr", "getSlots", (ins)>
    ];
}

#endif
