// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

func @test(%arg : !py.dynamic) -> i1 {
    %0 = py.bool.toI1 %arg
    return %0 : i1
}

// CHECK: @test
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[BITCAST:.*]] = llvm.bitcast %[[ARG]]
// CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[BITCAST]][0, 1]
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[GEP1]][0, 0]
// CHECK-NEXT: %[[USED:.*]] = llvm.load %[[GEP2]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[RESULT:.*]] = llvm.icmp "ne" %[[USED]], %[[ZERO]]
// CHECK-NEXT: llvm.return %[[RESULT]]
