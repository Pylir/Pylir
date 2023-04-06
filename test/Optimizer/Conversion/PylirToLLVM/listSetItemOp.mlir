// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.func @test(%arg : !py.dynamic, %index : index, %element : !py.dynamic) {
    list_setItem %arg[%index] to %element
    return
}

// CHECK: @test
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-SAME: %[[INDEX:[[:alnum:]]+]]
// CHECK-SAME: %[[ELEMENT:[[:alnum:]]+]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[ARG]][0, 2]
// CHECK-NEXT: %[[TUPLE_PTR:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[TRAILING:.*]] = llvm.getelementptr %[[TUPLE_PTR]][0, 2]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TRAILING]][0, %[[INDEX]]]
// CHECK-NEXT: llvm.store %[[ELEMENT]], %[[GEP]]
// CHECK-NEXT: llvm.return
