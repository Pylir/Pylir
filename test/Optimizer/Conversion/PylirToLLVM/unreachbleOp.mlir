// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.func @test(%arg : !py.dynamic) -> !py.dynamic {
    unreachable
}

// CHECK: @test
// CHECK-NEXT: llvm.unreachable
