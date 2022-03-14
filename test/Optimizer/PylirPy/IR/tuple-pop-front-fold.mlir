// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.tuple = #py.type<>

func @test() -> (!py.dynamic, !py.dynamic) {
    %0 = py.constant #py.tuple<value = (@builtins.tuple)>
    %element, %result = py.tuple.popFront %0
    return %element, %result : !py.dynamic, !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-DAG: %[[C1:.*]] = py.constant @builtins.tuple
// CHECK-DAG: %[[C2:.*]] = py.constant #py.tuple<value = ()>
// CHECK: return %[[C1]], %[[C2]]

func @test2(%arg0 : !py.dynamic) -> (!py.dynamic, !py.dynamic) {
    %0 = py.constant @builtins.tuple
    %1 = py.makeTuple (%0, *%arg0)
    %element, %result = py.tuple.popFront %1
    return %element, %result : !py.dynamic, !py.dynamic
}

// CHECK-LABEL: @test2
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-DAG: %[[C1:.*]] = py.constant @builtins.tuple
// CHECK-DAG: %[[C2:.*]] = py.makeTuple (*%[[ARG0]])
// CHECK: return %[[C1]], %[[C2]]
