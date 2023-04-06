// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.func @test(%arg : !py.dynamic) -> index {
    %0 = tuple_len %arg
    return %0 : index
}

// CHECK: @test
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[ARG]][0, 1]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.load %[[GEP]] {tbaa = [@tbaa::@"Python Tuple Size access"]}
// CHECK-NEXT: llvm.return %[[RESULT]]
