// RUN: pylir-opt %s | pylir-opt | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.dict = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.list = #py.type

// CHECK-LABEL: @makedictop_test
func @makedictop_test() -> !py.unknown {
    %0 = py.constant(#py.int<0>) : !py.unknown
    %1 = py.constant(#py.dict<{#py.str<"a"> to #py.int<3>, #py.str<"b"> to #py.list<[#py.int<5>]>}>) : !py.unknown
    %2 = py.constant(#py.str<"string">) : !py.unknown
    %3 = py.constant(#py.tuple<(#py.int<0>,#py.int<2>)>) : !py.unknown
    %4 = py.makeDict (%0 : %2,**%1,%2 : %3) : (!py.unknown, !py.unknown, !py.unknown), (!py.unknown, !py.unknown)
    py.return %4 : !py.class<@builtins.dict>
}

