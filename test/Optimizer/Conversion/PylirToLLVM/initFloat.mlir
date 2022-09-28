// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.float = #py.type
py.globalValue const @builtins.tuple = #py.type

func.func @foo(%value : f64) -> !py.dynamic {
    %0 = py.constant(#py.ref<@builtins.float>)
    %1 = pyMem.gcAllocObject %0
    %2 = pyMem.initFloat %1 to %value
    return %2 : !py.dynamic
}

// CHECK-LABEL: llvm.func @foo
// CHECK-SAME: %[[VALUE:[[:alnum:]]+]]
// CHECK: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc
// CHECK: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][0, 1]
// CHECK-NEXT: llvm.store %[[VALUE]], %[[GEP]]
// CHECK-NEXT: llvm.return %[[MEMORY]]
