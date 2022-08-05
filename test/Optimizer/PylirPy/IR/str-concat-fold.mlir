// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type

func.func @test() -> !py.dynamic {
    %0 = py.constant(#py.str<"Hello">)
    %1 = py.constant(#py.str<" ">)
    %2 = py.constant(#py.str<"World">)
    %3 = py.constant(#py.str<"!">)
    %4 = py.str.concat %0, %1, %2, %3
    return %4 : !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-DAG: %[[C1:.*]] = py.constant(#py.str<"Hello World!">)
// CHECK: return %[[C1]]
