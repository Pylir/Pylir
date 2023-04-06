// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.func @test(%arg0 : index, %arg1 : index, %arg2 : i1) -> !py.dynamic {
    %0 = int_fromUnsigned %arg0
    %1 = int_fromUnsigned %arg1
    %2 = arith.select %arg2, %0, %1 : !py.dynamic
    return %2 : !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG2:[[:alnum:]]+]]
// CHECK: %[[SEL:.*]] = arith.select %[[ARG2]], %[[ARG0]], %[[ARG1]]
// CHECK: %[[CONV:.*]] = int_fromUnsigned %[[SEL]]
// CHECK: return %[[CONV]]
