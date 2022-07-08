// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.tuple = #py.type

func.func @foo(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.constant(@builtins.str)
    %1 = pyMem.gcAllocObject %0
    %2 = pyMem.initStrFromInt %1 to %arg0
    return %2 : !py.dynamic
}

// CHECK-LABEL: @foo
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc(%{{.*}})
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][%[[ZERO]], 0]
// CHECK-NEXT: llvm.store %{{.*}}, %[[GEP]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[BUFFER:.*]] = llvm.getelementptr %[[MEMORY]][%[[ZERO]], 1]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[MP_INT_PTR:.*]] = llvm.getelementptr %[[ARG0]][%[[ZERO]], 1]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[SIZE_PTR:.*]] = llvm.getelementptr %[[BUFFER]][%[[ZERO]], 0]
// CHECK-NEXT: %[[TEN:.*]] = llvm.mlir.constant(10 : i32)
// CHECK-NEXT: llvm.call @mp_radix_size_overestimate(%[[MP_INT_PTR]], %[[TEN]], %[[SIZE_PTR]])
// CHECK-NEXT: %[[CAP:.*]] = llvm.load %[[SIZE_PTR]]
// CHECK-NEXT: %[[MEMORY2:.*]] = llvm.call @malloc(%[[CAP]])
// CHECK-NEXT: llvm.call @mp_to_radix(%[[MP_INT_PTR]], %[[MEMORY2]], %[[CAP]], %[[SIZE_PTR]], %[[TEN]])
// CHECK-NEXT: %[[SIZE:.*]] = llvm.load %[[SIZE_PTR]]
// CHECK-NEXT: %[[ONE_I:.*]] = llvm.mlir.constant(1 : index)
// CHECK-NEXT: %[[SIZE_1:.*]] = llvm.sub %[[SIZE]], %[[ONE_I]]
// CHECK-NEXT: llvm.store %[[SIZE_1]], %[[SIZE_PTR]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[CAP_PTR:.*]] = llvm.getelementptr %[[BUFFER]][%[[ZERO]], 1]
// CHECK-NEXT: llvm.store %[[CAP]], %[[CAP_PTR]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[BUFFER]][%[[ZERO]], 2]
// CHECK-NEXT: llvm.store %[[MEMORY2]], %[[GEP]]
// CHECK-NEXT: llvm.return %[[MEMORY]]
