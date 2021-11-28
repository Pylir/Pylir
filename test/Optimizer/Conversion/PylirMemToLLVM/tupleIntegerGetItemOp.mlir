// RUN: pylir-opt %s -convert-pylirMem-to-llvm --split-input-file | FileCheck %s

func @test(%arg : !py.dynamic, %index : index) -> !py.dynamic {
    %0 = py.tuple.integer.getItem %arg[%index : index]
    return %0 : !py.dynamic
}

// CHECK: @test
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-SAME: %[[INDEX:[[:alnum:]]+]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i32)
// CHECK-NEXT: %[[TWO:.*]] = llvm.mlir.constant(2 : i32)
// CHECK-NEXT: %[[BITCAST:.*]] = llvm.bitcast %[[ARG]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[BITCAST]][%[[ZERO]], %[[ONE]], %[[TWO]]]
// CHECK-NEXT: %[[BUFFER:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[BUFFER]][%[[INDEX]]]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: llvm.return %[[RESULT]]
