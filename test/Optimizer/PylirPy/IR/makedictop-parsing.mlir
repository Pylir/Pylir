// RUN: pylir-opt %s | pylir-opt | FileCheck %s

// CHECK-LABEL: @makedictop_test
func @makedictop_test() -> !py.dynamic {
    %0 = py.constant #py.int<0>
    %1 = py.constant #py.dict<{"a" to #py.int<3>, "b" to #py.list<[#py.int<5>]>}>
    %2 = py.constant "string"
    %3 = py.constant #py.tuple<(#py.int<0>,#py.int<2>)>
    %4 = py.makeDict (%0 : %2,**%1,%2 : %3)
    return %4 : !py.dynamic
}

