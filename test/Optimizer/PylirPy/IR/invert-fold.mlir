// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

// CHECK-LABEL: @invert_integer
// CHECK: %[[RES:.*]] = py.constant #py.int<-6>
// CHECK: return %[[RES]]
func @invert_integer() -> !py.dynamic {
    %0 = py.constant #py.int<5>
    %1 = py.invert %0
    return %1 : !py.dynamic
}
