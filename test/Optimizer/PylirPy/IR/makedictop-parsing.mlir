// RUN: pylir-opt %s | pylir-opt | FileCheck %s

// CHECK-LABEL: @makedictop_test
py.func @makedictop_test(%hash: index) -> !py.dynamic {
    %0 = constant(#py.int<0>)
    %1 = constant(#py.dict<{#py.str<"a"> to #py.int<3>, #py.str<"b"> to #py.list<[#py.int<5>]>}>)
    %2 = constant(#py.str<"string">)
    %3 = constant(#py.tuple<(#py.int<0>,#py.int<2>)>)
    %4 = makeDict (%0 hash(%hash) : %2,**%1,%2 hash(%hash) : %3)
    return %4 : !py.dynamic
}

