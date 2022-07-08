// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.int = #py.type
py.globalValue const @builtins.tuple = #py.type

func.func @foo(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = py.constant(@builtins.int)
    %1 = pyMem.gcAllocObject %0
    %2 = pyMem.initIntAdd %1 to %arg0 + %arg1
    return %2 : !py.dynamic
}

// CHECK-LABEL: llvm.func @foo
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[RESULT_INT:.*]] = llvm.getelementptr %[[MEMORY]][%[[ZERO]], 1]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[LHS_INT:.*]] = llvm.getelementptr %[[ARG0]][%[[ZERO]], 1]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[RHS_INT:.*]] = llvm.getelementptr %[[ARG1]][%[[ZERO]], 1]
// CHECK-NEXT: llvm.call @mp_init(%[[RESULT_INT]])
// CHECK-NEXT: llvm.call @mp_add(%[[LHS_INT]], %[[RHS_INT]], %[[RESULT_INT]])
// CHECK-NEXT: llvm.return %[[MEMORY]]
