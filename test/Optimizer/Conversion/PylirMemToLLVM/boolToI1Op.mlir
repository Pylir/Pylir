// RUN: pylir-opt %s -convert-pylirMem-to-llvm --split-input-file | FileCheck %s

func @test(%arg : !py.dynamic) -> i1 {
    %0 = py.bool.toI1 %arg
    return %0 : i1
}

// CHECK: @test
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i32)
// CHECK-NEXT: %[[BITCAST:.*]] = llvm.bitcast %[[ARG]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[BITCAST]][%[[ZERO]], %[[ONE]], %[[ZERO]]]
// CHECK-NEXT: %[[USED:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// CHECK-NEXT: %[[RESULT:.*]] = llvm.icmp "ne" %[[USED]], %[[ZERO]]
// CHECK-NEXT: llvm.return %[[RESULT]]
