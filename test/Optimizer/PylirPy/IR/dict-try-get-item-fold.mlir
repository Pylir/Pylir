// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.dict = #py.type

func @test(%arg0 : !py.dynamic) -> i1 {
    %0 = py.constant(#py.dict<{}>)
    %2, %3 = py.dict.tryGetItem %0[%arg0]
    return %3 : i1
}

// CHECK-LABEL: @test
// CHECK-DAG: %[[C1:.*]] = arith.constant false
// CHECK: return %[[C1]]
