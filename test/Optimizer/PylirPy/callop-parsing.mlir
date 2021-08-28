// RUN: pylir-opt %s | pylir-opt | FileCheck %s

// CHECK-LABEL: @callop_test
func @callop_test() -> !py.dynamic {
    %0 = py.constant @outside
    %1 = py.constant #py.dict<{"a" to 3, "b" to #py.list<[5]>}>
    %2 = py.constant "string"
    %3 = py.constant #py.tuple<(0,2)>
    %4 = py.call %0(*%3, %2, **%1)
    %5 = py.call %0()
    return %4 : !py.dynamic
}
