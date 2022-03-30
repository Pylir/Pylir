// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

func @test(%arg : !py.unknown, %index : index) -> !py.unknown {
    %0 = py.tuple.getItem %arg[%index] : (!py.unknown) -> !py.unknown
    return %0 : !py.unknown
}

// CHECK: @test
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-SAME: %[[INDEX:[[:alnum:]]+]]
// CHECK-NEXT: %[[BITCAST:.*]] = llvm.bitcast %[[ARG]]
// CHECK-NEXT: %[[TRAILING:.*]] = llvm.getelementptr %[[BITCAST]][0, 2]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TRAILING]][0, %[[INDEX]]]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: llvm.return %[[RESULT]]
