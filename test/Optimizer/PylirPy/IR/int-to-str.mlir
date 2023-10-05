// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

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
