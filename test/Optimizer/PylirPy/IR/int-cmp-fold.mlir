// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

func @test1(%arg0 : !py.dynamic) -> i1 {
    %0 = py.constant #py.int<5>
    %1 = py.int.cmp eq %0, %arg0
    return %1 : i1
}

// CHECK-LABEL: @test1
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = py.constant #py.int<5>
// CHECK-NEXT: %[[RESULT:.*]] = py.int.cmp eq %[[ARG0]], %[[C]]
// CHECK-NEXT: return %[[RESULT]]

func @test2(%arg0 : !py.dynamic) -> i1 {
    %0 = py.constant #py.int<5>
    %1 = py.int.cmp ne %0, %arg0
    return %1 : i1
}

// CHECK-LABEL: @test2
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = py.constant #py.int<5>
// CHECK-NEXT: %[[RESULT:.*]] = py.int.cmp ne %[[ARG0]], %[[C]]
// CHECK-NEXT: return %[[RESULT]]
