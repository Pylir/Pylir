// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

func @constant_tuple() -> index {
    %0 = py.constant #py.tuple<(#py.int<0>, "text", 5.0)>
    %1 = py.tuple.len %0 : index
    return %1 : index
}

// CHECK-LABEL: @constant_tuple
// CHECK: %[[RESULT:.*]] = arith.constant 3 : index
// CHECK: return %[[RESULT]]

// -----
