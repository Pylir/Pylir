// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.tuple = #py.type<>
py.globalValue @builtins.str = #py.type<>

func @test() -> !py.dynamic {
    %0 = py.constant #py.tuple<value = ()>
    %1 = py.constant @builtins.tuple
    %2 = py.tuple.prepend %1, %0
    return %2 : !py.dynamic
}

// CHECK-LABEL: @test
// CHECK: %[[C:.*]] = py.constant #py.tuple<value = (@builtins.tuple)>
// CHECK: return %[[C]]

func @test2(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeTuple (*%arg0)
    %1 = py.constant @builtins.tuple
    %2 = py.tuple.prepend %1, %0
    return %2 : !py.dynamic
}

// CHECK-LABEL: @test2
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[C:.*]] = py.constant @builtins.tuple
// CHECK: %[[RESULT:.*]] = py.makeTuple (%[[C]], *%[[ARG0]])
// CHECK: return %[[RESULT]]

func @test3(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.constant #py.tuple<value = (#py.str<value = "value">)>
    %2 = py.tuple.prepend %arg0, %0
    return %2 : !py.dynamic
}

// CHECK-LABEL: @test3
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-DAG: %[[C:.*]] = py.constant #py.str<value = "value">
// CHECK: %[[RESULT:.*]] = py.makeTuple (%[[ARG0]], %[[C]])
// CHECK: return %[[RESULT]]
