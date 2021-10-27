// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

// CHECK-LABEL: @neg_floats
// CHECK: %[[RES:.*]] = py.constant -5.0
// CHECK: return %[[RES]]
func @neg_floats() -> !py.dynamic {
    %0 = py.constant 5.0
    %1 = py.neg %0
    return %1 : !py.dynamic
}

// -----

// CHECK-LABEL: @neg_integer
// CHECK: %[[RES:.*]] = py.constant #py.int<-5>
// CHECK: return %[[RES]]
func @neg_integer() -> !py.dynamic {
    %0 = py.constant #py.int<5>
    %1 = py.neg %0
    return %1 : !py.dynamic
}
