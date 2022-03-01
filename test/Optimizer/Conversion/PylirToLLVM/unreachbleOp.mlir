// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

func @test(%arg : !py.dynamic) -> !py.dynamic {
    py.unreachable
}

// CHECK: @test
// CHECK-NEXT: llvm.unreachable
