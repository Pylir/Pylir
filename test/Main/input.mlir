// RUN: pylir %s -o - -S -emit-llvm | FileCheck %s

py.globalValue const @const$ = #py.tuple<value = (#py.str<value = "__slots__">)>
py.globalValue @builtins.type = #py.type<slots = {__slots__ = @const$}>
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.function = #py.type
py.globalValue @builtins.dict = #py.type

func @foo(%arg0 : !py.unknown) -> !py.unknown {
    %0 = py.typeOf %arg0 : (!py.unknown) -> !py.unknown
    %1 = py.typeOf %0 : (!py.unknown) -> !py.unknown
    %2 = py.getSlot "__slots__" from %0 : %1 : (!py.unknown, !py.unknown) -> !py.unknown
    return %2 : !py.unknown
}

// CHECK-LABEL: define %PyObject addrspace({{[0-9]+}})* @foo(%PyObject addrspace({{[0-9]+}})* %{{.*}})
