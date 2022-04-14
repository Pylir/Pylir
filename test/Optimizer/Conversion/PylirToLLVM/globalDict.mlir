// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.object = #py.type
py.globalValue @builtins.dict = #py.type
py.globalValue @builtins.str = #py.type

func @test() -> !py.dynamic {
    %0 = py.constant(#py.dict<{#py.str<"Hello"> to #py.str<"World">}>)
    return %0 : !py.dynamic
}

// CHECK-LABEL: llvm.func internal @"$__GLOBAL_INIT__"
// CHECK-NEXT: %[[ADDRESS:.*]] = llvm.mlir.addressof
// CHECK-NEXT: %[[DICT:.*]] = llvm.bitcast %[[ADDRESS]]
// CHECK-NEXT: %[[KEY:.*]] = llvm.mlir.addressof
// CHECK-NEXT: %[[KEY_CAST:.*]] = llvm.bitcast %[[KEY]]
// CHECK-NEXT: %[[VALUE:.*]] = llvm.mlir.addressof
// CHECK-NEXT: %[[VALUE_CAST:.*]] = llvm.bitcast %[[VALUE]]
// CHECK-NEXT: llvm.call @pylir_dict_insert(%[[DICT]], %[[KEY_CAST]], %[[VALUE_CAST]])

// CHECK: llvm.mlir.global_ctors {ctors = [@"$__GLOBAL_INIT__"], priorities = [{{.*}}]}
