// RUN: pylir-opt %s -convert-pylirMem-to-llvm --split-input-file | FileCheck %s

func @test(%arg : !py.dynamic, %index : index) -> !py.dynamic {
    %0 = py.tuple.getItem %arg[%index]
    return %0 : !py.dynamic
}

// CHECK: @test
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-SAME: %[[INDEX:[[:alnum:]]+]]
// CHECK-NEXT: %[[BITCAST:.*]] = llvm.bitcast %[[ARG]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[BITCAST]][0, 1, 2]
// CHECK-NEXT: %[[BUFFER:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[BUFFER]][%[[INDEX]]]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: llvm.return %[[RESULT]]
