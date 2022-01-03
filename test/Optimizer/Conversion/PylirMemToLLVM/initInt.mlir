// RUN: pylir-opt %s -convert-pylirMem-to-llvm --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.int = #py.type

func @foo(%value : i64) -> !py.dynamic {
    %0 = py.constant @builtins.int
    %1 = pyMem.gcAllocObject %0
    %2 = pyMem.initInt %1 to %value : i64
    return %2 : !py.dynamic
}

// CHECK-LABEL: llvm.func @foo
// CHECK-SAME: %[[VALUE:[[:alnum:]]+]]
// CHECK: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc
// CHECK: %[[CASTED:.*]] = llvm.bitcast %[[MEMORY]]
// CHECK: %[[INTEGER:.*]] = llvm.bitcast %[[CASTED]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i32)
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[INTEGER]][%[[ZERO]], %[[ONE]]]
// CHECK-NEXT: llvm.call @mp_init_u64(%[[GEP]], %[[VALUE]])
// CHECK-NEXT: llvm.return %[[CASTED]]
