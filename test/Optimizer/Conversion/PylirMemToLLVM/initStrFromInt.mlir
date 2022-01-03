// RUN: pylir-opt %s -convert-pylirMem-to-llvm --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.str = #py.type

func @foo(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.constant @builtins.str
    %1 = pyMem.gcAllocObject %0
    %2 = pyMem.initStrFromInt %1 to %arg0
    return %2 : !py.dynamic
}

// CHECK-LABEL: @foo
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc(%{{.*}})
// CHECK: %[[RESULT:.*]] = llvm.bitcast %[[MEMORY]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[RESULT]][%[[ZERO]], %[[ZERO]]]
// CHECK-NEXT: llvm.store %{{.*}}, %[[GEP]]
// CHECK-NEXT: %[[STRING:.*]] = llvm.bitcast %[[RESULT]]
// CHECK-NEXT: %[[INT:.*]] = llvm.bitcast %[[ARG0]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i32)
// CHECK-NEXT: %[[MP_INT_PTR:.*]] = llvm.getelementptr %[[INT]][%[[ZERO]], %[[ONE]]]
// CHECK-NEXT: %[[SIZE_PTR:.*]] = llvm.getelementptr %[[STRING]][%[[ZERO]], %[[ONE]], %[[ZERO]]]
// CHECK-NEXT: %[[TEN:.*]] = llvm.mlir.constant(10 : i32)
// CHECK-NEXT: llvm.call @mp_radix_size_overestimate(%[[MP_INT_PTR]], %[[TEN]], %[[SIZE_PTR]])
// CHECK-NEXT: %[[CAP:.*]] = llvm.load %[[SIZE_PTR]]
// CHECK-NEXT: %[[MEMORY:.*]] = llvm.call @malloc(%[[CAP]])
// CHECK-NEXT: llvm.call @mp_to_radix(%[[MP_INT_PTR]], %[[MEMORY]], %[[CAP]], %[[SIZE_PTR]], %[[TEN]])
// CHECK-NEXT: %[[SIZE:.*]] = llvm.load %[[SIZE_PTR]]
// CHECK-NEXT: %[[ONE_I:.*]] = llvm.mlir.constant(1 : index)
// CHECK-NEXT: %[[SIZE_1:.*]] = llvm.sub %[[SIZE]], %[[ONE_I]]
// CHECK-NEXT: llvm.store %[[SIZE_1]], %[[SIZE_PTR]]
// CHECK-NEXT: %[[CAP_PTR:.*]] = llvm.getelementptr %[[STRING]][%[[ZERO]], %[[ONE]], %[[ONE]]]
// CHECK-NEXT: llvm.store %[[CAP]], %[[CAP_PTR]]
// CHECK-NEXT: %[[TWO:.*]] = llvm.mlir.constant(2 : i32)
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[STRING]][%[[ZERO]], %[[ONE]], %[[TWO]]]
// CHECK-NEXT: llvm.store %[[MEMORY]], %[[GEP]]
// CHECK-NEXT: llvm.return %[[RESULT]]
