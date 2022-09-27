// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.str = #py.type

func.func @test() -> (!py.dynamic, !py.dynamic) {
    %0 = py.constant(#py.int<5>)
    %1 = py.constant(#py.int<-3>)
    %2 = py.int.toStr %0
    %3 = py.int.toStr %1
    return %2, %3 : !py.dynamic, !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-DAG: %[[C1:.*]] = py.constant(#py.str<"5">)
// CHECK-DAG: %[[C2:.*]] = py.constant(#py.str<"-3">)
// CHECK-NEXT: return %[[C1]], %[[C2]]
