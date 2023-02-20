// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

func.func @test(%arg : !py.dynamic) -> !py.dynamic {
    %0 = py.type.mro %arg
    return %0 : !py.dynamic
}

// CHECK: @test
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[MRO:.*]] = llvm.getelementptr %[[ARG]][0, 3]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.load %[[MRO]] {tbaa = [@tbaa::@"Python Type MRO access"]}
// CHECK-NEXT: llvm.return %[[RESULT]]
