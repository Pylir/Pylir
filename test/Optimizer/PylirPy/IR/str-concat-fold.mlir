// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str

py.func @test() -> !py.dynamic {
    %0 = constant(#py.str<"Hello">)
    %1 = constant(#py.str<" ">)
    %2 = constant(#py.str<"World">)
    %3 = constant(#py.str<"!">)
    %4 = str_concat %0, %1, %2, %3
    return %4 : !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-DAG: %[[C1:.*]] = constant(#py.str<"Hello World!">)
// CHECK: return %[[C1]]
