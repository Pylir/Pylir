
#ifndef PYLIR_PY_TRAITS_TABLEGEN
#define PYLIR_PY_TRAITS_TABLEGEN

def AlwaysBound : NativeOpTrait<"AlwaysBound"> {
    let cppNamespace = "pylir::Py";
}

def NoCapture : NativeOpTrait<"NoCapture"> {
    let cppNamespace = "pylir::Py";
}

#endif
