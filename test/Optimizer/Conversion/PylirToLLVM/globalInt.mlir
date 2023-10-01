// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_object = #py.globalValue<builtins.object, initializer = #py.type>
py.external @builtins.object, #builtins_object
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple

py.func @test() -> !py.dynamic {
    %0 = constant(#py.int<5>)
    return %0 : !py.dynamic
}

// CHECK: llvm.call @mp_init(%[[MP_INT_PTR:[[:alnum:]]+]])
// CHECK: llvm.call @mp_unpack
// CHECK-SAME: %[[MP_INT_PTR]]
// CHECK-SAME-6: %{{[[:alnum:]]+}}

// CHECK: llvm.mlir.global_ctors {ctors = [@{{.*}}], priorities = [{{.*}}]}
