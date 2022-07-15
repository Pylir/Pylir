// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

func.func @test(%arg : !py.dynamic, %index : index) -> !py.dynamic {
    %0 = py.list.getItem %arg[%index]
    return %0 : !py.dynamic
}

// CHECK: @test
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-SAME: %[[INDEX:[[:alnum:]]+]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[ARG]][%[[ZERO]], 2]
// CHECK-NEXT: %[[TUPLE_PTR:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[TRAILING:.*]] = llvm.getelementptr %[[TUPLE_PTR]][%[[ZERO]], 2]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TRAILING]][%[[ZERO]], %[[INDEX]]]
// CHECK-NEXT: %[[ELEMENT:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: llvm.return %[[ELEMENT]]
