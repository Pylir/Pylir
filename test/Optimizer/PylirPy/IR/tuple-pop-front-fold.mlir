// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type

func @test() -> (!py.unknown, !py.unknown) {
    %0 = py.constant(#py.tuple<value = (@builtins.tuple)>) : !py.unknown
    %element, %result = py.tuple.popFront %0 : (!py.unknown) -> (!py.unknown, !py.unknown)
    return %element, %result : !py.unknown, !py.unknown
}

// CHECK-LABEL: @test
// CHECK-DAG: %[[C1:.*]] = py.constant(@builtins.tuple)
// CHECK-DAG: %[[C2:.*]] = py.constant(#py.tuple<value = ()>)
// CHECK: return %[[C1]], %[[C2]]

func @test2(%arg0 : !py.unknown) -> (!py.unknown, !py.unknown) {
    %0 = py.constant(@builtins.tuple) : !py.unknown
    %1 = py.makeTuple (%0, *%arg0) : (!py.unknown, !py.unknown) -> !py.unknown
    %element, %result = py.tuple.popFront %1 : (!py.unknown) -> (!py.unknown, !py.unknown)
    return %element, %result : !py.unknown, !py.unknown
}

// CHECK-LABEL: @test2
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-DAG: %[[C1:.*]] = py.constant(@builtins.tuple)
// CHECK-DAG: %[[C2:.*]] = py.makeTuple (*%[[ARG0]])
// CHECK: return %[[C1]], %[[C2]]
