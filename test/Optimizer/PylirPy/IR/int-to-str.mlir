// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str

py.func @test() -> (!py.dynamic, !py.dynamic) {
    %0 = constant(#py.int<5>)
    %1 = constant(#py.int<-3>)
    %2 = int_toStr %0
    %3 = int_toStr %1
    return %2, %3 : !py.dynamic, !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-DAG: %[[C1:.*]] = constant(#py.str<"5">)
// CHECK-DAG: %[[C2:.*]] = constant(#py.str<"-3">)
// CHECK-NEXT: return %[[C1]], %[[C2]]
