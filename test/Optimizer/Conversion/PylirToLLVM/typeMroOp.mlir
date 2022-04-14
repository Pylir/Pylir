// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

func @test(%arg : !py.dynamic) -> !py.dynamic {
    %0 = py.type.mro %arg
    return %0 : !py.dynamic
}

// CHECK: @test
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[BITCAST:.*]] = llvm.bitcast %[[ARG]]
// CHECK-NEXT: %[[MRO:.*]] = llvm.getelementptr %[[BITCAST]][0, 3]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.load %[[MRO]]
// CHECK-NEXT: llvm.return %[[RESULT]]
