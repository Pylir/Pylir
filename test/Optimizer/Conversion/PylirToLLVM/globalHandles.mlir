// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.globalValue @builtins.tuple = #py.tuple<()>

py.globalHandle @handle

func @test() -> !py.dynamic {
    %0 = py.constant(#py.tuple<()>)
    py.store %0 into @handle
    %1 = py.load @handle
    return %1 : !py.dynamic
}

// CHECK: llvm.mlir.global external @handle()
// CHECK-NEXT: %[[NULL:.*]] = llvm.mlir.null
// CHECK-NEXT: llvm.return %[[NULL]]

// CHECK-LABEL: @test
// CHECK-NEXT: %[[CONST:.*]] = llvm.mlir.addressof @{{const\$.*}}
// CHECK-NEXT: %[[CAST:.*]] = llvm.bitcast %[[CONST]]
// CHECK-NEXT: %[[HANDLE:.*]] = llvm.mlir.addressof @handle
// CHECK-NEXT: llvm.store %[[CAST]], %[[HANDLE]]
// CHECK-NEXT: %[[HANDLE:.*]] = llvm.mlir.addressof @handle
// CHECK-NEXT: %[[VALUE:.*]] = llvm.load %[[HANDLE]]
// CHECK-NEXT: llvm.return %[[VALUE]]
