// RUN: pylir-opt %s -convert-arith-to-llvm -convert-pylir-to-llvm --reconcile-unrealized-casts --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.int = #py.type
py.globalValue const @builtins.tuple = #py.type

py.func @foo(%value : index) -> !py.dynamic {
    %0 = constant(#py.ref<@builtins.int>)
    %c0 = arith.constant 0 : index
    %1 = pyMem.gcAllocObject %0[%c0]
    %2 = pyMem.initIntUnsigned %1 to %value
    return %2 : !py.dynamic
}

// CHECK-LABEL: llvm.func @foo
// CHECK-SAME: %[[VALUE:[[:alnum:]]+]]
// CHECK: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc
// CHECK: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][0, 1]
// CHECK-NEXT: llvm.call @mp_init_u64(%[[GEP]], %[[VALUE]])
// CHECK-NEXT: llvm.return %[[MEMORY]]

py.func @bar(%value : index) -> !py.dynamic {
    %0 = constant(#py.ref<@builtins.int>)
     %c0 = arith.constant 0 : index
    %1 = pyMem.gcAllocObject %0[%c0]
    %2 = pyMem.initIntSigned %1 to %value
    return %2 : !py.dynamic
}

// CHECK-LABEL: llvm.func @bar
// CHECK-SAME: %[[VALUE:[[:alnum:]]+]]
// CHECK: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc
// CHECK: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][0, 1]
// CHECK-NEXT: llvm.call @mp_init_i64(%[[GEP]], %[[VALUE]])
// CHECK-NEXT: llvm.return %[[MEMORY]]
