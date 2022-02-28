
#ifndef PYLIR_PY_TRAITS_TABLEGEN
#define PYLIR_PY_TRAITS_TABLEGEN

def AlwaysBound : NativeOpTrait<"AlwaysBound"> {
    let cppNamespace = "pylir::Py";
}

def NoCapture : NativeOpTrait<"NoCapture"> {
    let cppNamespace = "pylir::Py";
}

class RuntimeType<string name> : NativeOpTrait<name # "RuntimeType"> {
    let cppNamespace = "pylir::Py";
}

def ExceptionHandling : NativeOpTrait<"ExceptionHandling"> {
    let cppNamespace = "pylir::Py";
}

#endif
