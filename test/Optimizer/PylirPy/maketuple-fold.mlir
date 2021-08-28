// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

// CHECK-LABEL: @maketuple_simple
// CHECK: %[[RES:.*]] = py.constant #py.tuple
// CHECK-SAME: "text"
// CHECK-SAME: 2
// CHECK-SAME: 3.0
// CHECK: return %[[RES]]
func @maketuple_simple() -> !py.dynamic {
    %0 = py.constant "text"
    %1 = py.constant 2
    %2 = py.constant 3.0
    %3 = py.makeTuple ( %0, %1, %2 )
    return %3 : !py.dynamic
}

// -----

// CHECK-LABEL: @maketuple_expansion
// CHECK: %[[RES:.*]] = py.constant #py.tuple
// CHECK-SAME: "text"
// CHECK-SAME: 2
// CHECK-SAME: 3.0
// CHECK: return %[[RES]]
func @maketuple_expansion() -> !py.dynamic {
    %0 = py.constant "text"
    %1 = py.constant #py.list<[2, 3.0]>
    %2 = py.makeTuple ( %0, *%1 )
    return %2 : !py.dynamic
}
