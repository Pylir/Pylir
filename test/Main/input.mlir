// RUN: pylir %s -o - -S -emit-llvm | FileCheck %s

py.globalValue const @const$ = #py.tuple<(#py.str<"__slots__">)>
py.globalValue @builtins.type = #py.type<slots: #py.slots<{"__slots__" to @const$}>>
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.function = #py.type
py.globalValue @builtins.dict = #py.type

func @foo(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.typeOf %arg0
    %1 = py.typeOf %0
    %2 = py.getSlot "__slots__" from %0 : %1
    return %2 : !py.dynamic
}

// CHECK-LABEL: define %PyObject addrspace({{[0-9]+}})* @foo(%PyObject addrspace({{[0-9]+}})* %{{.*}})
