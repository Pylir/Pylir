// RUN: pylir-opt %s -cse --split-input-file | FileCheck %s

py.func @test(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> (!py.dynamic, !py.dynamic) {
    %0 = py.tuple.copy %arg0 : %arg1
    %1 = py.tuple.copy %arg0 : %arg1
    return %0, %1 : !py.dynamic, !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-NEXT: %[[FIRST:.*]] = py.tuple.copy %[[ARG0]] : %[[ARG1]]
// CHECK-NEXT: %[[SECOND:.*]] = py.tuple.copy %[[ARG0]] : %[[ARG1]]
// CHECK-NEXT: return %[[FIRST]], %[[SECOND]]
