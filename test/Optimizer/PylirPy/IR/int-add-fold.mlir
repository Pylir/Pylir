// RUN: pylir-opt %s -canonicalize | FileCheck %s

py.func @test1(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = constant(#py.int<5>)
    %1 = constant(#py.int<3>)
    %2 = int_add %arg0, %0
    %3 = int_add %2, %1
    return %3 : !py.dynamic
}

// CHECK-LABEL: @test1
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = constant(#py.int<8>)
// CHECK-NEXT: %[[RESULT:.*]] = int_add %[[ARG0]], %[[C]]
// CHECK-NEXT: return %[[RESULT]]

py.func @test2(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = constant(#py.int<5>)
    %1 = constant(#py.int<3>)
    %2 = int_add %arg0, %0
    %3 = int_add %arg1, %1
    %4 = int_add %2, %3
    return %4 : !py.dynamic
}

// CHECK-LABEL: @test2
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = constant(#py.int<8>)
// CHECK-NEXT: %[[SUM:.*]] = int_add %[[ARG0]], %[[ARG1]]
// CHECK-NEXT: %[[RESULT:.*]] = int_add %[[SUM]], %[[C]]
// CHECK-NEXT: return %[[RESULT]]
