// RUN: pylir-opt %s -convert-arith-to-llvm -convert-pylir-to-llvm --reconcile-unrealized-casts --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.int = #py.type
py.globalValue const @builtins.tuple = #py.type

py.func @foo(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = constant(#py.ref<@builtins.int>)
    %c0 = arith.constant 0 : index
    %1 = pyMem.gcAllocObject %0[%c0]
    %2 = pyMem.initIntAdd %1 to %arg0 + %arg1
    return %2 : !py.dynamic
}

// CHECK-LABEL: llvm.func @foo
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc
// CHECK: %[[RESULT_INT:.*]] = llvm.getelementptr %[[MEMORY]][0, 1]
// CHECK-NEXT: %[[LHS_INT:.*]] = llvm.getelementptr %[[ARG0]][0, 1]
// CHECK-NEXT: %[[RHS_INT:.*]] = llvm.getelementptr %[[ARG1]][0, 1]
// CHECK-NEXT: llvm.call @mp_init(%[[RESULT_INT]])
// CHECK-NEXT: llvm.call @mp_add(%[[LHS_INT]], %[[RHS_INT]], %[[RESULT_INT]])
// CHECK-NEXT: llvm.return %[[MEMORY]]
