// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.global @handle : !py.dynamic

py.func @test() -> i1 {
    %0 = load @handle : !py.dynamic
    %1 = isUnboundValue %0
    return %1 : i1
}

// CHECK-LABEL: @test
// CHECK-NEXT: %[[HANDLE:.*]] = llvm.mlir.addressof @handle
// CHECK-NEXT: %[[VALUE:.*]] = llvm.load %[[HANDLE]]
// CHECK-NEXT: %[[NULL:.*]] = llvm.mlir.zero
// CHECK-NEXT: %[[RESULT:.*]] = llvm.icmp "eq" %[[VALUE]], %[[NULL]]
// CHECK-NEXT: llvm.return %[[RESULT]]
