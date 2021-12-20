// RUN: pylir-opt %s -convert-pylirMem-to-llvm --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

func @test() -> !py.dynamic {
    %0 = py.constant #py.int<5>
    return %0 : !py.dynamic
}

// CHECK: llvm.call @mp_init(%[[MP_INT_PTR:[[:alnum:]]+]])
// CHECK: llvm.call @mp_unpack
// CHECK-SAME: %[[MP_INT_PTR]]
// CHECK-SAME-6: %{{[[:alnum:]]+}}

// CHECK: llvm.mlir.global_ctors {ctors = [@{{.*}}], priorities = [{{.*}}]}
