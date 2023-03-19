// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.dict = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.float = #py.type
py.globalValue @builtins.str = #py.type

py.func @test(%arg0 : !py.dynamic, %arg1 : index) -> i1 {
    %0 = constant(#py.dict<{}>)
    %2 = py.dict.tryGetItem %0[%arg0 hash(%arg1)]
    %3 = isUnboundValue %2
    return %3 : i1
}

// CHECK-LABEL: @test
// CHECK-DAG: %[[C1:.*]] = arith.constant true
// CHECK: return %[[C1]]

py.func @test2(%arg0 : index) -> !py.dynamic {
    %0 = constant(#py.dict<{#py.int<5> to #py.str<"value">}>)
    %1 = constant(#py.float<5.0>)
    %2 = py.dict.tryGetItem %0[%1 hash(%arg0)]
    return %2 : !py.dynamic
}

// CHECK-LABEL: @test2
// CHECK: %[[C:.*]] = constant(#py.str<"value">)
// CHECK: return %[[C]]
