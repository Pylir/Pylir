// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type

py.func @test() -> !py.dynamic {
    %0 = constant(#py.tuple<()>)
    %1 = constant(#py.ref<@builtins.tuple>)
    %2 = py.tuple.prepend %1, %0
    return %2 : !py.dynamic
}

// CHECK-LABEL: @test
// CHECK: %[[C:.*]] = constant(#py.tuple<(#py.ref<@builtins.tuple>)>)
// CHECK: return %[[C]]

py.func @test3(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = constant(#py.tuple<(#py.str<"value">)>)
    %2 = py.tuple.prepend %arg0, %0
    return %2 : !py.dynamic
}

// CHECK-LABEL: @test3
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-DAG: %[[C:.*]] = constant(#py.str<"value">)
// CHECK: %[[RESULT:.*]] = makeTuple (%[[ARG0]], %[[C]])
// CHECK: return %[[RESULT]]
