
#ifndef PYLIR_PY_TRAITS_TABLEGEN
#define PYLIR_PY_TRAITS_TABLEGEN

def AlwaysBound : NativeOpTrait<"AlwaysBound"> {
    let cppNamespace = "pylir::Py";
}

def NoCapture : NativeOpTrait<"NoCapture"> {
    let cppNamespace = "pylir::Py";
}

class RefinedType<string type> : NativeOpTrait<type # "RefinedType"> {
	let cppNamespace = "pylir::Py";
}

#endif
