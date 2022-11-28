// RUN: pylir-opt %s --canonicalize | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.float = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type

func.func @test1(%hash: index) -> !py.dynamic {
    %0 = py.constant(#py.int<0>)
    %1 = py.constant(#py.float<0.0>)
    %2 = py.constant(#py.str<"string">)
    %3 = py.constant(#py.tuple<(#py.int<0>,#py.int<2>)>)
    %4 = py.makeDict (%0 hash(%hash) : %2, %2 hash(%hash) : %2, %1 hash(%hash) : %3)
    return %4 : !py.dynamic
}

// CHECK-LABEL: @test1
// CHECK-SAME: %[[HASH:[[:alnum:]]+]]: index
// CHECK-DAG: %[[F:.*]] = py.constant(#py.float<0.{{0+}}{{(e(\+|-)0+)?}}>)
// CHECK-DAG: %[[S:.*]] = py.constant(#py.str<"string">
// CHECK-DAG: %[[T:.*]] = py.constant(#py.tuple<(#py.int<0>, #py.int<2>)>)
// CHECK: %[[D:.*]] = py.makeDict (%[[S]] hash(%[[HASH]]) : %[[S]], %[[F]] hash(%[[HASH]]) : %[[T]])
// CHECK-NEXT: return %[[D]]

func.func @test2(%hash: index, %key: !py.dynamic) -> !py.dynamic {
    %2 = py.constant(#py.str<"string">)
    %3 = py.constant(#py.tuple<(#py.int<0>,#py.int<2>)>)
    %4 = py.makeDict (%key hash(%hash) : %2, %2 hash(%hash) : %2, %key hash(%hash) : %3)
    return %4 : !py.dynamic
}

// CHECK-LABEL: @test2
// CHECK-SAME: %[[HASH:[[:alnum:]]+]]: index
// CHECK-SAME: %[[KEY:[[:alnum:]]+]]
// CHECK-DAG: %[[S:.*]] = py.constant(#py.str<"string">
// CHECK-DAG: %[[T:.*]] = py.constant(#py.tuple<(#py.int<0>, #py.int<2>)>)
// CHECK: %[[D:.*]] = py.makeDict (%[[S]] hash(%[[HASH]]) : %[[S]], %[[KEY]] hash(%[[HASH]]) : %[[T]])
// CHECK-NEXT: return %[[D]]