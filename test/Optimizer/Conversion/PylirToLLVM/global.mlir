// RUN: pylir-opt %s -convert-arith-to-llvm -convert-pylir-to-llvm='target-triple=x86_64-unknown-linux-gnu' --reconcile-unrealized-casts --split-input-file | FileCheck %s --check-prefixes=CHECK,SECTION_COFF
// RUN: pylir-opt %s -convert-arith-to-llvm -convert-pylir-to-llvm='target-triple=x86_64-apple-darwin21.6.0' --reconcile-unrealized-casts --split-input-file | FileCheck %s --check-prefixes=CHECK,SECTION_MACHO

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple

py.global @handle : !py.dynamic

py.func @test() -> !py.dynamic {
    %0 = constant(#py.tuple<()>)
    store %0 : !py.dynamic into @handle
    %1 = load @handle : !py.dynamic
    return %1 : !py.dynamic
}

// CHECK-DAG: #[[DESC:.*]] = #llvm.tbaa_type_desc<id = "Python Handle"{{.*}}>
// CHECK-DAG: #[[$PYTHON_HANDLE:.*]] = #llvm.tbaa_tag<base_type = #[[DESC]], access_type = #[[DESC]], offset = 0>

// CHECK-LABEL: llvm.mlir.global external @handle()
// SECTION_COFF-SAME: section = "py_root"
// SECTION_MACHO-SAME: section = "__DATA,py_root"
// CHECK-NEXT: %[[UNDEF:.*]] = llvm.mlir.undef
// CHECK-NEXT: llvm.return %[[UNDEF]]

// CHECK-LABEL: @test
// CHECK-NEXT: %[[CONST:.*]] = llvm.mlir.addressof @{{const\$.*}}
// CHECK-NEXT: %[[HANDLE:.*]] = llvm.mlir.addressof @handle
// CHECK-NEXT: llvm.store %[[CONST]], %[[HANDLE]] {tbaa = [#[[$PYTHON_HANDLE]]]}
// CHECK-NEXT: %[[HANDLE:.*]] = llvm.mlir.addressof @handle
// CHECK-NEXT: %[[VALUE:.*]] = llvm.load %[[HANDLE]] {tbaa = [#[[$PYTHON_HANDLE]]]}
// CHECK-NEXT: llvm.return %[[VALUE]]

// -----

py.global @handle : index

py.func @test() -> index {
    %0 = arith.constant 5 : index
    store %0 : index into @handle
    %1 = load @handle : index
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

py.func @test() -> index {
    %0 = arith.constant 5 : index
    store %0 : index into @handle
    %1 = load @handle : index
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

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_object = #py.globalValue<builtins.object, initializer = #py.type>
py.external @builtins.object, #builtins_object
#builtins_dict = #py.globalValue<builtins.dict, initializer = #py.type>
py.external @builtins.dict, #builtins_dict
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str

py.global @handle : !py.dynamic = #py.dict<{}>

py.func @test() -> !py.dynamic {
    %0 = constant(#py.tuple<()>)
    store %0 : !py.dynamic into @handle
    %1 = load @handle : !py.dynamic
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
