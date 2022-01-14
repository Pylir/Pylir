// RUN: pylir-opt %s -convert-pylirMem-to-llvm --split-input-file | FileCheck %s

func @test(%arg : !py.dynamic) -> index {
    %0 = py.tuple.len %arg
    return %0 : index
}

// CHECK: @test
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[BITCAST:.*]] = llvm.bitcast %[[ARG]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[BITCAST]][0, 1, 0]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: llvm.return %[[RESULT]]
