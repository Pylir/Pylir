// RUN: pylir-opt %s -canonicalize | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

func @test1(%arg0 : !py.unknown) -> !py.class<@builtins.int> {
    %0 = py.constant(#py.int<5>) : !py.unknown
    %1 = py.constant(#py.int<3>) : !py.unknown
    %2 = py.int.add %arg0, %0 : !py.unknown, !py.unknown
    %3 = py.int.add %2, %1 : !py.class<@builtins.int>, !py.unknown
    return %3 : !py.class<@builtins.int>
}

// CHECK-LABEL: @test1
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = py.constant(#py.int<8>)
// CHECK-NEXT: %[[RESULT:.*]] = py.int.add %[[ARG0]], %[[C]]
// CHECK-NEXT: return %[[RESULT]]

func @test2(%arg0 : !py.unknown, %arg1 : !py.unknown) -> !py.class<@builtins.int> {
    %0 = py.constant(#py.int<5>) : !py.unknown
    %1 = py.constant(#py.int<3>) : !py.unknown
    %2 = py.int.add %arg0, %0 : !py.unknown, !py.unknown
    %3 = py.int.add %arg1, %1 : !py.unknown, !py.unknown
    %4 = py.int.add %2, %3 : !py.class<@builtins.int>, !py.class<@builtins.int>
    return %4 : !py.class<@builtins.int>
}

// CHECK-LABEL: @test2
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = py.constant(#py.int<8>)
// CHECK-NEXT: %[[SUM:.*]] = py.int.add %[[ARG0]], %[[ARG1]]
// CHECK-NEXT: %[[RESULT:.*]] = py.int.add %[[SUM]], %[[C]]
// CHECK-NEXT: return %[[RESULT]]
