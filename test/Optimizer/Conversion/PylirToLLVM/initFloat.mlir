// RUN: pylir-opt %s -convert-arith-to-llvm -convert-pylir-to-llvm --reconcile-unrealized-casts --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.float = #py.type
py.globalValue const @builtins.tuple = #py.type

func.func @foo(%value : f64) -> !py.dynamic {
    %0 = py.constant(#py.ref<@builtins.float>)
    %c0 = arith.constant 0 : index
    %1 = pyMem.gcAllocObject %0[%c0]
    %2 = pyMem.initFloat %1 to %value
    return %2 : !py.dynamic
}

// CHECK-LABEL: llvm.func @foo
// CHECK-SAME: %[[VALUE:[[:alnum:]]+]]
// CHECK: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc
// CHECK: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][0, 1]
// CHECK-NEXT: llvm.store %[[VALUE]], %[[GEP]] {tbaa = [@tbaa::@"Python Float Value access"]}
// CHECK-NEXT: llvm.return %[[MEMORY]]
