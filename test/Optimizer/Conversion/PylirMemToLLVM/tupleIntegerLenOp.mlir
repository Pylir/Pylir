// RUN: pylir-opt %s -convert-pylirMem-to-llvm --split-input-file | FileCheck %s

func @test(%arg : !py.dynamic) -> index {
    %0 = py.tuple.integer.len %arg : index
    return %0 : index
}

// CHECK: @test
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i32)
// CHECK-NEXT: %[[BITCAST:.*]] = llvm.bitcast %[[ARG]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[BITCAST]][%[[ZERO]], %[[ONE]], %[[ZERO]]]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: llvm.return %[[RESULT]]
