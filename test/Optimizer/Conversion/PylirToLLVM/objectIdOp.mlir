// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.func @test(%arg0 : !py.dynamic) -> index {
    %0 = object_id %arg0
    return %0 : index
}

// CHECK: @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.ptrtoint %[[ARG0]]
// CHECK-NEXT: llvm.return %[[RESULT]]
