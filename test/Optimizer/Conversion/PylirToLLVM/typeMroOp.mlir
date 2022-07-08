// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

func.func @test(%arg : !py.dynamic) -> !py.dynamic {
    %0 = py.type.mro %arg
    return %0 : !py.dynamic
}

// CHECK: @test
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[MRO:.*]] = llvm.getelementptr %[[ARG]][%[[ZERO]], 3]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.load %[[MRO]]
// CHECK-NEXT: llvm.return %[[RESULT]]
