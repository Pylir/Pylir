// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

// CHECK-LABEL: @pos_floats
// CHECK: %[[RES:.*]] = py.constant 5.0
// CHECK: return %[[RES]]
func @pos_floats() -> !py.dynamic {
    %0 = py.constant 5.0
    %1 = py.pos %0
    return %1 : !py.dynamic
}

// -----

// CHECK-LABEL: @pos_integer
// CHECK: %[[RES:.*]] = py.constant #py.int<5>
// CHECK: return %[[RES]]
func @pos_integer() -> !py.dynamic {
    %0 = py.constant #py.int<5>
    %1 = py.pos %0
    return %1 : !py.dynamic
}
