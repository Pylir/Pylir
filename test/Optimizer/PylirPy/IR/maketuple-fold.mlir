// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

// CHECK-LABEL: @maketuple_simple
// CHECK: %[[RES:.*]] = constant(#py.tuple
// CHECK-SAME: #py.str<"text">
// CHECK-SAME: #py.int<2>
// CHECK-SAME: #py.float<3.000000e+00>
// CHECK: return %[[RES]]
py.func @maketuple_simple() -> !py.dynamic {
    %0 = constant(#py.str<"text">)
    %1 = constant(#py.int<2>)
    %2 = constant(#py.float<3.0>)
    %3 = makeTuple ( %0, %1, %2 )
    return %3 : !py.dynamic
}

// -----

// CHECK-LABEL: @maketuple_expansion
// CHECK: %[[RES:.*]] = constant(#py.tuple
// CHECK-SAME: #py.str<"text">
// CHECK-SAME: #py.int<2>
// CHECK-SAME: #py.float<3.000000e+00>
// CHECK: return %[[RES]]
py.func @maketuple_expansion() -> !py.dynamic {
    %0 = constant(#py.str<"text">)
    %1 = constant(#py.list<[#py.int<2>, #py.float<3.0>]>)
    %2 = makeTuple ( %0, *%1 )
    return %2 : !py.dynamic
}
