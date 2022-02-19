// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

func @test(%arg : !py.dynamic) -> index {
    %0 = py.dict.len %arg
    return %0 : index
}

// CHECK: @test
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[BITCAST:.*]] = llvm.bitcast %[[ARG]]
// CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[BITCAST]][0, 1]
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[GEP1]][0, 0]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.load %[[GEP2]]
// CHECK-NEXT: llvm.return %[[RESULT]]
