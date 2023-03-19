// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type

py.func @test() -> !py.dynamic {
    %0 = constant(#py.str<"Hello">)
    %1 = constant(#py.str<" ">)
    %2 = constant(#py.str<"World">)
    %3 = constant(#py.str<"!">)
    %4 = py.str.concat %0, %1, %2, %3
    return %4 : !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-DAG: %[[C1:.*]] = constant(#py.str<"Hello World!">)
// CHECK: return %[[C1]]
