// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type

func @test() -> !py.unknown {
    %0 = py.constant(#py.tuple<()>) : !py.unknown
    %1 = py.constant(@builtins.tuple) : !py.unknown
    %2 = py.tuple.prepend %1, %0 : (!py.unknown, !py.unknown) -> !py.unknown
    return %2 : !py.unknown
}

// CHECK-LABEL: @test
// CHECK: %[[C:.*]] = py.constant(#py.tuple<(@builtins.tuple)>)
// CHECK: return %[[C]]

func @test3(%arg0 : !py.unknown) -> !py.unknown {
    %0 = py.constant(#py.tuple<(#py.str<"value">)>) : !py.unknown
    %2 = py.tuple.prepend %arg0, %0 : (!py.unknown, !py.unknown) -> !py.unknown
    return %2 : !py.unknown
}

// CHECK-LABEL: @test3
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-DAG: %[[C:.*]] = py.constant(#py.str<"value">)
// CHECK: %[[RESULT:.*]] = py.makeTuple (%[[ARG0]], %[[C]])
// CHECK: return %[[RESULT]]
