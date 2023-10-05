// RUN: pylir-opt %s --canonicalize | FileCheck %s

py.func @test1(%hash: index) -> !py.dynamic {
    %0 = constant(#py.int<0>)
    %1 = constant(#py.float<0.0>)
    %2 = constant(#py.str<"string">)
    %3 = constant(#py.tuple<(#py.int<0>,#py.int<2>)>)
    %4 = makeDict (%0 hash(%hash) : %2, %2 hash(%hash) : %2, %1 hash(%hash) : %3)
    return %4 : !py.dynamic
}

// CHECK-LABEL: @test1
// CHECK-SAME: %[[HASH:[[:alnum:]]+]]: index
// CHECK-DAG: %[[F:.*]] = constant(#py.float<0.{{0+}}{{(e(\+|-)0+)?}}>)
// CHECK-DAG: %[[S:.*]] = constant(#py.str<"string">
// CHECK-DAG: %[[T:.*]] = constant(#py.tuple<(#py.int<0>, #py.int<2>)>)
// CHECK: %[[D:.*]] = makeDict (%[[S]] hash(%[[HASH]]) : %[[S]], %[[F]] hash(%[[HASH]]) : %[[T]])
// CHECK-NEXT: return %[[D]]

py.func @test2(%hash: index, %key: !py.dynamic) -> !py.dynamic {
    %2 = constant(#py.str<"string">)
    %3 = constant(#py.tuple<(#py.int<0>,#py.int<2>)>)
    %4 = makeDict (%key hash(%hash) : %2, %2 hash(%hash) : %2, %key hash(%hash) : %3)
    return %4 : !py.dynamic
}

// CHECK-LABEL: @test2
// CHECK-SAME: %[[HASH:[[:alnum:]]+]]: index
// CHECK-SAME: %[[KEY:[[:alnum:]]+]]
// CHECK-DAG: %[[S:.*]] = constant(#py.str<"string">
// CHECK-DAG: %[[T:.*]] = constant(#py.tuple<(#py.int<0>, #py.int<2>)>)
// CHECK: %[[D:.*]] = makeDict (%[[S]] hash(%[[HASH]]) : %[[S]], %[[KEY]] hash(%[[HASH]]) : %[[T]])
// CHECK-NEXT: return %[[D]]
