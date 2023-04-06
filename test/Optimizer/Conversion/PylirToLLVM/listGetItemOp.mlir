// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.func @test(%arg : !py.dynamic, %index : index) -> !py.dynamic {
    %0 = list_getItem %arg[%index]
    return %0 : !py.dynamic
}

// CHECK: @test
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-SAME: %[[INDEX:[[:alnum:]]+]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[ARG]][0, 2]
// CHECK-NEXT: %[[TUPLE_PTR:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[TRAILING:.*]] = llvm.getelementptr %[[TUPLE_PTR]][0, 2]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TRAILING]][0, %[[INDEX]]]
// CHECK-NEXT: %[[ELEMENT:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: llvm.return %[[ELEMENT]]
