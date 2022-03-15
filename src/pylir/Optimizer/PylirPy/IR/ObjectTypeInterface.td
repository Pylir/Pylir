#ifndef PYLIRPY_INTERFACES_OBJECT_TYPE_INTERFACE
#define PYLIRPY_INTERFACES_OBJECT_TYPE_INTERFACE

include "mlir/IR/OpBase.td"

def ObjectTypeInterface : TypeInterface<"ObjectTypeInterface"> {
    let cppNamespace = "::pylir::Py";

    let methods = [
        InterfaceMethod<[{}], "::mlir::FlatSymbolRefAttr", "getTypeObject", (ins)>,
        InterfaceMethod<[{}], "::llvm::ArrayRef<std::pair<mlir::StringAttr, pylir::Py::ObjectTypeInterface>>",
                                                                                            "getSlots", (ins)>,
    ];
}

def ObejctTypeInterfaceC : Type<CPred<"$_self.isa<::pylir::Py::ObjectTypeInterface>()">, "ObjectTypeInterface instance",
	"::pylir::Py::ObjectTypeInterface">;

#endif
