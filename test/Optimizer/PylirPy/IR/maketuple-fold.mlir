// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.float = #py.type
py.globalValue @builtins.tuple = #py.type

// CHECK-LABEL: @maketuple_simple
// CHECK: %[[RES:.*]] = py.constant(#py.tuple
// CHECK-SAME: #py.str<value = "text">
// CHECK-SAME: #py.int<value = 2>
// CHECK-SAME: #py.float<value = 3.000000e+00>
// CHECK: return %[[RES]]
func @maketuple_simple() -> !py.unknown {
    %0 = py.constant(#py.str<value = "text">) : !py.unknown
    %1 = py.constant(#py.int<value = 2>) : !py.unknown
    %2 = py.constant(#py.float<value = 3.0>) : !py.unknown
    %3 = py.makeTuple ( %0, %1, %2 ) : (!py.unknown, !py.unknown, !py.unknown) -> !py.unknown
    return %3 : !py.unknown
}

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.float = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.list = #py.type

// CHECK-LABEL: @maketuple_expansion
// CHECK: %[[RES:.*]] = py.constant(#py.tuple
// CHECK-SAME: #py.str<value = "text">
// CHECK-SAME: #py.int<value = 2>
// CHECK-SAME: #py.float<value = 3.000000e+00>
// CHECK: return %[[RES]]
func @maketuple_expansion() -> !py.unknown {
    %0 = py.constant(#py.str<value = "text">) : !py.unknown
    %1 = py.constant(#py.list<value = [#py.int<value = 2>, #py.float<value = 3.0>]>) : !py.unknown
    %2 = py.makeTuple ( %0, *%1 ) : (!py.unknown, !py.unknown) -> !py.unknown
    return %2 : !py.unknown
}
