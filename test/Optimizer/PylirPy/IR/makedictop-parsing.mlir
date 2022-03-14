// RUN: pylir-opt %s | pylir-opt | FileCheck %s

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.int = #py.type<>
py.globalValue @builtins.dict = #py.type<>
py.globalValue @builtins.str = #py.type<>
py.globalValue @builtins.tuple = #py.type<>
py.globalValue @builtins.list = #py.type<>

// CHECK-LABEL: @makedictop_test
func @makedictop_test() -> !py.dynamic {
    %0 = py.constant #py.int<value = 0>
    %1 = py.constant #py.dict<value = {#py.str<value = "a"> to #py.int<value = 3>, #py.str<value = "b"> to #py.list<value = [#py.int<value = 5>]>}>
    %2 = py.constant #py.str<value = "string">
    %3 = py.constant #py.tuple<value = (#py.int<value = 0>,#py.int<value = 2>)>
    %4 = py.makeDict (%0 : %2,**%1,%2 : %3)
    return %4 : !py.dynamic
}

