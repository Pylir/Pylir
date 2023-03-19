// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.func @test(%arg : !py.dynamic) -> !py.dynamic {
    %0 = py.type.slots %arg
    return %0 : !py.dynamic
}

// CHECK: @test
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[MRO:.*]] = llvm.getelementptr %[[ARG]][0, 4]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.load %[[MRO]] {tbaa = [@tbaa::@"Python Type Instance Slots access"]}
// CHECK-NEXT: llvm.return %[[RESULT]]
