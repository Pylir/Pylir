// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.func @test() -> index {
    %0 = constant(#py.dict<{#py.str<"test"> to #py.int<3>}>)
    %2 = dict_len %0
    return %2 : index
}

// CHECK-LABEL: @test
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK: return %[[C1]]
