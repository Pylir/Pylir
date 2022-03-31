// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.globalValue @builtins.tuple = #py.tuple<()>

py.globalHandle @handle

func @test() -> !py.unknown {
    %0 = py.constant(#py.tuple<()>) : !py.unknown
    py.store %0 into @handle : !py.unknown
    %1 = py.load @handle : !py.unknown
    return %1 : !py.unknown
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
