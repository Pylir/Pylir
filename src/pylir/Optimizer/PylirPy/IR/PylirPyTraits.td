
#ifndef PYLIR_PY_TRAITS_TABLEGEN
#define PYLIR_PY_TRAITS_TABLEGEN

include "pylir/Optimizer/PylirPy/Interfaces/ObjectFromTypeObjectInterface.td"

def AlwaysBound : NativeOpTrait<"AlwaysBound"> {
    let cppNamespace = "pylir::Py";
}

def NoCapture : NativeOpTrait<"NoCapture"> {
    let cppNamespace = "pylir::Py";
}

class RefinedType<string type> : NativeOpTrait<type # "RefinedType"> {
	let cppNamespace = "pylir::Py";
}

def RefinedObjectFromTypeObjectImpl : NativeOpTrait<"RefinedObjectFromTypeObject"> {
	let cppNamespace = "pylir::Py";
}

def RefinedObjectFromTypeObject : TraitList<[RefinedObjectFromTypeObjectImpl, ObjectFromTypeObjectInterface]>;

#endif
