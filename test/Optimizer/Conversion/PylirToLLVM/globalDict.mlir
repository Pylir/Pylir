// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.object = #py.type
py.globalValue @builtins.dict = #py.type
py.globalValue @builtins.str = #py.type

func.func @test() -> !py.dynamic {
    %0 = py.constant(#py.dict<{#py.str<"Hello"> to #py.str<"World">}>)
    return %0 : !py.dynamic
}

// CHECK-LABEL: llvm.func internal @"$__GLOBAL_INIT__"
// CHECK-NEXT: %[[DICT:.*]] = llvm.mlir.addressof
// CHECK-NEXT: %[[KEY:.*]] = llvm.mlir.addressof
// CHECK-NEXT: %[[HASH:.*]] = llvm.call @pylir_str_hash(%[[KEY]])
// CHECK-NEXT: %[[VALUE:.*]] = llvm.mlir.addressof
// CHECK-NEXT: llvm.call @pylir_dict_insert_unique(%[[DICT]], %[[KEY]], %[[HASH]], %[[VALUE]])

// CHECK: llvm.mlir.global_ctors {ctors = [@"$__GLOBAL_INIT__"], priorities = [{{.*}}]}
