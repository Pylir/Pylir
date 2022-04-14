// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.float = #py.type
py.globalValue @builtins.tuple = #py.type

// CHECK-LABEL: @maketuple_simple
// CHECK: %[[RES:.*]] = py.constant(#py.tuple
// CHECK-SAME: #py.str<"text">
// CHECK-SAME: #py.int<2>
// CHECK-SAME: #py.float<3.000000e+00>
// CHECK: return %[[RES]]
func @maketuple_simple() -> !py.dynamic {
    %0 = py.constant(#py.str<"text">)
    %1 = py.constant(#py.int<2>)
    %2 = py.constant(#py.float<3.0>)
    %3 = py.makeTuple ( %0, %1, %2 )
    return %3 : !py.dynamic
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
// CHECK-SAME: #py.str<"text">
// CHECK-SAME: #py.int<2>
// CHECK-SAME: #py.float<3.000000e+00>
// CHECK: return %[[RES]]
func @maketuple_expansion() -> !py.dynamic {
    %0 = py.constant(#py.str<"text">)
    %1 = py.constant(#py.list<[#py.int<2>, #py.float<3.0>]>)
    %2 = py.makeTuple ( %0, *%1 )
    return %2 : !py.dynamic
}
