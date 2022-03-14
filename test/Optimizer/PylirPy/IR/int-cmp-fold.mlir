// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.int = #py.type<>

func @test1(%arg0 : !py.dynamic) -> i1 {
    %0 = py.constant #py.int<value = 5>
    %1 = py.int.cmp eq %0, %arg0
    return %1 : i1
}

// CHECK-LABEL: @test1
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = py.constant #py.int<value = 5>
// CHECK-NEXT: %[[RESULT:.*]] = py.int.cmp eq %[[ARG0]], %[[C]]
// CHECK-NEXT: return %[[RESULT]]

func @test2(%arg0 : !py.dynamic) -> i1 {
    %0 = py.constant #py.int<value = 5>
    %1 = py.int.cmp ne %0, %arg0
    return %1 : i1
}

// CHECK-LABEL: @test2
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = py.constant #py.int<value = 5>
// CHECK-NEXT: %[[RESULT:.*]] = py.int.cmp ne %[[ARG0]], %[[C]]
// CHECK-NEXT: return %[[RESULT]]

func @test3(%arg0 : !py.dynamic) -> i1 {
    %0 = py.constant #py.int<value = 5>
    %1 = arith.constant 1 : i1
    %2 = py.int.cmp ne %0, %arg0
    %3 = arith.xori %1, %2 : i1
    return %3 : i1
}

// CHECK-LABEL: @test3
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = py.constant #py.int<value = 5>
// CHECK-NEXT: %[[RESULT:.*]] = py.int.cmp eq %[[ARG0]], %[[C]]
// CHECK-NEXT: return %[[RESULT]]

func @test4(%arg0 : !py.dynamic) -> i1 {
    %0 = py.constant #py.int<value = 5>
    %1 = arith.constant 1 : i1
    %2 = py.int.cmp eq %0, %arg0
    %3 = arith.xori %1, %2 : i1
    return %3 : i1
}

// CHECK-LABEL: @test4
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = py.constant #py.int<value = 5>
// CHECK-NEXT: %[[RESULT:.*]] = py.int.cmp ne %[[ARG0]], %[[C]]
// CHECK-NEXT: return %[[RESULT]]

func @test5(%arg0 : !py.dynamic) -> i1 {
    %0 = py.constant #py.int<value = 5>
    %1 = arith.constant 1 : i1
    %2 = py.int.cmp lt %0, %arg0
    %3 = arith.xori %1, %2 : i1
    return %3 : i1
}

// CHECK-LABEL: @test5
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = py.constant #py.int<value = 5>
// CHECK-NEXT: %[[RESULT:.*]] = py.int.cmp le %[[ARG0]], %[[C]]
// CHECK-NEXT: return %[[RESULT]]


func @test6(%arg0 : !py.dynamic) -> i1 {
    %0 = py.constant #py.int<value = 5>
    %1 = arith.constant 1 : i1
    %2 = py.int.cmp le %0, %arg0
    %3 = arith.xori %1, %2 : i1
    return %3 : i1
}

// CHECK-LABEL: @test6
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = py.constant #py.int<value = 5>
// CHECK-NEXT: %[[RESULT:.*]] = py.int.cmp lt %[[ARG0]], %[[C]]
// CHECK-NEXT: return %[[RESULT]]

func @test7(%arg0 : !py.dynamic) -> i1 {
    %0 = py.constant #py.int<value = 5>
    %1 = arith.constant 1 : i1
    %2 = py.int.cmp gt %0, %arg0
    %3 = arith.xori %1, %2 : i1
    return %3 : i1
}

// CHECK-LABEL: @test7
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = py.constant #py.int<value = 5>
// CHECK-NEXT: %[[RESULT:.*]] = py.int.cmp ge %[[ARG0]], %[[C]]
// CHECK-NEXT: return %[[RESULT]]


func @test8(%arg0 : !py.dynamic) -> i1 {
    %0 = py.constant #py.int<value = 5>
    %1 = arith.constant 1 : i1
    %2 = py.int.cmp ge %0, %arg0
    %3 = arith.xori %1, %2 : i1
    return %3 : i1
}

// CHECK-LABEL: @test8
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = py.constant #py.int<value = 5>
// CHECK-NEXT: %[[RESULT:.*]] = py.int.cmp gt %[[ARG0]], %[[C]]
// CHECK-NEXT: return %[[RESULT]]
