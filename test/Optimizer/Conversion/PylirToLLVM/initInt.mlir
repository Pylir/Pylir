// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.int = #py.type
py.globalValue const @builtins.tuple = #py.type

func @foo(%value : i64) -> !py.unknown {
    %0 = py.constant(@builtins.int) : !py.unknown
    %1 = pyMem.gcAllocObject %0 : !py.unknown
    %2 = pyMem.initInt %1 to %value : (i64) -> !py.unknown
    return %2 : !py.unknown
}

// CHECK-LABEL: llvm.func @foo
// CHECK-SAME: %[[VALUE:[[:alnum:]]+]]
// CHECK: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc
// CHECK: %[[CASTED:.*]] = llvm.bitcast %[[MEMORY]]
// CHECK: %[[INTEGER:.*]] = llvm.bitcast %[[CASTED]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[INTEGER]][0, 1]
// CHECK-NEXT: llvm.call @mp_init_u64(%[[GEP]], %[[VALUE]])
// CHECK-NEXT: llvm.return %[[CASTED]]
