// RUN: pylir-opt %s -convert-arith-to-llvm -convert-pylir-to-llvm='target-triple=x86_64-unknown-linux-gnu' --reconcile-unrealized-casts --split-input-file | FileCheck %s --check-prefixes=CHECK,SECTION_COFF
// RUN: pylir-opt %s -convert-arith-to-llvm -convert-pylir-to-llvm='target-triple=x86_64-apple-darwin21.6.0' --reconcile-unrealized-casts --split-input-file | FileCheck %s --check-prefixes=CHECK,SECTION_MACHO

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type

py.global @handle : !py.dynamic

func.func @test() -> !py.dynamic {
    %0 = py.constant(#py.tuple<()>)
    py.store %0 : !py.dynamic into @handle
    %1 = py.load @handle : !py.dynamic
    return %1 : !py.dynamic
}

// CHECK: llvm.mlir.global external @handle()
// SECTION_COFF-SAME: section = "py_root"
// SECTION_MACHO-SAME: section = "__DATA,py_root"
// CHECK-NEXT: %[[UNDEF:.*]] = llvm.mlir.undef
// CHECK-NEXT: llvm.return %[[UNDEF]]

// CHECK-LABEL: @test
// CHECK-NEXT: %[[CONST:.*]] = llvm.mlir.addressof @{{const\$.*}}
// CHECK-NEXT: %[[HANDLE:.*]] = llvm.mlir.addressof @handle
// CHECK-NEXT: llvm.store %[[CONST]], %[[HANDLE]]
// CHECK-NEXT: %[[HANDLE:.*]] = llvm.mlir.addressof @handle
// CHECK-NEXT: %[[VALUE:.*]] = llvm.load %[[HANDLE]]
// CHECK-NEXT: llvm.return %[[VALUE]]

// -----

py.global @handle : index

func.func @test() -> index {
    %0 = arith.constant 5 : index
    py.store %0 : index into @handle
    %1 = py.load @handle : index
    return %1 : index
}

// CHECK: llvm.mlir.global external @handle()
// CHECK-NOT: section
// CHECK-NEXT: %[[UNDEF:.*]] = llvm.mlir.undef
// CHECK-NEXT: llvm.return %[[UNDEF]]

// CHECK-LABEL: @test
// CHECK-NEXT: %[[CONST:.*]] = llvm.mlir.constant(5 : i{{.*}})
// CHECK-NEXT: %[[HANDLE:.*]] = llvm.mlir.addressof @handle
// CHECK-NEXT: llvm.store %[[CONST]], %[[HANDLE]]
// CHECK-NEXT: %[[HANDLE:.*]] = llvm.mlir.addressof @handle
// CHECK-NEXT: %[[VALUE:.*]] = llvm.load %[[HANDLE]]
// CHECK-NEXT: llvm.return %[[VALUE]]

// -----

py.global @handle : index = 3 : index

func.func @test() -> index {
    %0 = arith.constant 5 : index
    py.store %0 : index into @handle
    %1 = py.load @handle : index
    return %1 : index
}

// CHECK: llvm.mlir.global external @handle()
// CHECK-NOT: section
// CHECK-NEXT: %[[C:.*]] = llvm.mlir.constant(3 : i{{.*}})
// CHECK-NEXT: llvm.return %[[C]]

// CHECK-LABEL: @test
// CHECK-NEXT: %[[CONST:.*]] = llvm.mlir.constant(5 : i{{.*}})
// CHECK-NEXT: %[[HANDLE:.*]] = llvm.mlir.addressof @handle
// CHECK-NEXT: llvm.store %[[CONST]], %[[HANDLE]]
// CHECK-NEXT: %[[HANDLE:.*]] = llvm.mlir.addressof @handle
// CHECK-NEXT: %[[VALUE:.*]] = llvm.load %[[HANDLE]]
// CHECK-NEXT: llvm.return %[[VALUE]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.object = #py.type
py.globalValue @builtins.dict = #py.type
py.globalValue @builtins.str = #py.type

py.global @handle : !py.dynamic = #py.dict<{}>

func.func @test() -> !py.dynamic {
    %0 = py.constant(#py.tuple<()>)
    py.store %0 : !py.dynamic into @handle
    %1 = py.load @handle : !py.dynamic
    return %1 : !py.dynamic
}

// CHECK: llvm.mlir.global external @handle()
// SECTION_COFF-SAME: section = "py_root"
// SECTION_MACHO-SAME: section = "__DATA,py_root"
// CHECK-NEXT: %[[REF:.*]] = llvm.mlir.addressof @{{const\$.*}}
// CHECK-NEXT: llvm.return %[[REF]]

// CHECK-LABEL: @test
// CHECK-NEXT: %[[CONST:.*]] = llvm.mlir.addressof @{{const\$.*}}
// CHECK-NEXT: %[[HANDLE:.*]] = llvm.mlir.addressof @handle
// CHECK-NEXT: llvm.store %[[CONST]], %[[HANDLE]]
// CHECK-NEXT: %[[HANDLE:.*]] = llvm.mlir.addressof @handle
// CHECK-NEXT: %[[VALUE:.*]] = llvm.load %[[HANDLE]]
// CHECK-NEXT: llvm.return %[[VALUE]]
