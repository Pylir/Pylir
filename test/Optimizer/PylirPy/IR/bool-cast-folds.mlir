// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.func @test(%value : !py.dynamic) -> !py.dynamic {
    %0 = py.bool.toI1 %value
    %1 = py.bool.fromI1 %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: py.func @test
// CHECK-SAME: %[[VALUE:[[:alnum:]]+]]
// CHECK: return %[[VALUE]]

py.func @test2(%value : i1) -> i1 {
    %0 = py.bool.fromI1 %value
    %1 = py.bool.toI1 %0
    return %1 : i1
}

// CHECK-LABEL: py.func @test2
// CHECK-SAME: %[[VALUE:[[:alnum:]]+]]
// CHECK: return %[[VALUE]]
