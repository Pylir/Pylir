// RUN: pylir-opt %s -convert-pylir-to-llvm='target-triple=x86_64-unknown-linux-gnu' --split-input-file | FileCheck %s --check-prefixes=CHECK,SECTION_COFF
// RUN: pylir-opt %s -convert-pylir-to-llvm='target-triple=x86_64-apple-darwin21.6.0' --split-input-file | FileCheck %s --check-prefixes=CHECK,SECTION_MACHO

py.globalValue @builtins.tuple = #py.tuple<()>

py.globalHandle @handle

func.func @test() -> !py.dynamic {
    %0 = py.constant(#py.tuple<()>)
    py.store %0 into @handle
    %1 = py.load @handle
    return %1 : !py.dynamic
}

// CHECK: llvm.mlir.global external @handle()
// SECTION_COFF-SAME: section = "py_root"
// SECTION_MACHO-SAME: section = "__DATA,py_root"
// CHECK-NEXT: %[[NULL:.*]] = llvm.mlir.null
// CHECK-NEXT: llvm.return %[[NULL]]

// CHECK-LABEL: @test
// CHECK-NEXT: %[[CONST:.*]] = llvm.mlir.addressof @{{const\$.*}}
// CHECK-NEXT: %[[HANDLE:.*]] = llvm.mlir.addressof @handle
// CHECK-NEXT: llvm.store %[[CONST]], %[[HANDLE]]
// CHECK-NEXT: %[[HANDLE:.*]] = llvm.mlir.addressof @handle
// CHECK-NEXT: %[[VALUE:.*]] = llvm.load %[[HANDLE]]
// CHECK-NEXT: llvm.return %[[VALUE]]
