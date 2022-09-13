// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.dict = #py.type

func.func @test(%arg0 : !py.dynamic, %arg1 : index) -> i1 {
    %0 = py.constant(#py.dict<{}>)
    %2 = py.dict.tryGetItem %0[%arg0 hash(%arg1)]
    %3 = py.isUnboundValue %2
    return %3 : i1
}

// CHECK-LABEL: @test
// CHECK-DAG: %[[C1:.*]] = arith.constant true
// CHECK: return %[[C1]]
