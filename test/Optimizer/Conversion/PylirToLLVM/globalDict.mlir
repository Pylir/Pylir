// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

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

py.func @test() -> !py.dynamic {
    %0 = constant(#py.dict<{#py.str<"Hello"> to #py.str<"World">}>)
    return %0 : !py.dynamic
}

// CHECK-LABEL: llvm.func internal @"$__GLOBAL_INIT__"
// CHECK-NEXT: %[[DICT:.*]] = llvm.mlir.addressof
// CHECK-NEXT: %[[KEY:.*]] = llvm.mlir.addressof
// CHECK-NEXT: %[[HASH:.*]] = llvm.call @pylir_str_hash(%[[KEY]])
// CHECK-NEXT: %[[VALUE:.*]] = llvm.mlir.addressof
// CHECK-NEXT: llvm.call @pylir_dict_insert_unique(%[[DICT]], %[[KEY]], %[[HASH]], %[[VALUE]])

// CHECK: llvm.mlir.global_ctors {ctors = [@"$__GLOBAL_INIT__"], priorities = [{{.*}}]}
