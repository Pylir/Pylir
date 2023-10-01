// RUN: pylir-opt %s -convert-arith-to-llvm -convert-pylir-to-llvm --reconcile-unrealized-casts --split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple

py.func @foo(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = constant(#builtins_str)
    %c0 = arith.constant 0 : index
    %1 = pyMem.gcAllocObject %0[%c0]
    %2 = pyMem.initStrFromInt %1 to %arg0
    return %2 : !py.dynamic
}

// CHECK-LABEL: @foo
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc(%{{.*}})
// CHECK: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][0, 0]
// CHECK-NEXT: llvm.store %{{.*}}, %[[GEP]]
// CHECK-NEXT: %[[BUFFER:.*]] = llvm.getelementptr %[[MEMORY]][0, 1]
// CHECK-NEXT: %[[MP_INT_PTR:.*]] = llvm.getelementptr %[[ARG0]][0, 1]
// CHECK-NEXT: %[[SIZE_PTR:.*]] = llvm.getelementptr %[[BUFFER]][0, 0]
// CHECK-NEXT: %[[TEN:.*]] = llvm.mlir.constant(10 : i32)
// CHECK-NEXT: llvm.call @mp_radix_size_overestimate(%[[MP_INT_PTR]], %[[TEN]], %[[SIZE_PTR]])
// CHECK-NEXT: %[[CAP:.*]] = llvm.load %[[SIZE_PTR]]
// CHECK-NEXT: %[[MEMORY2:.*]] = llvm.call @malloc(%[[CAP]])
// CHECK-NEXT: llvm.call @mp_to_radix(%[[MP_INT_PTR]], %[[MEMORY2]], %[[CAP]], %[[SIZE_PTR]], %[[TEN]])
// CHECK-NEXT: %[[SIZE:.*]] = llvm.load %[[SIZE_PTR]]
// CHECK-NEXT: %[[ONE_I:.*]] = llvm.mlir.constant(1 : index)
// CHECK-NEXT: %[[SIZE_1:.*]] = llvm.sub %[[SIZE]], %[[ONE_I]]
// CHECK-NEXT: llvm.store %[[SIZE_1]], %[[SIZE_PTR]]
// CHECK-NEXT: %[[CAP_PTR:.*]] = llvm.getelementptr %[[BUFFER]][0, 1]
// CHECK-NEXT: llvm.store %[[CAP]], %[[CAP_PTR]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[BUFFER]][0, 2]
// CHECK-NEXT: llvm.store %[[MEMORY2]], %[[GEP]]
// CHECK-NEXT: llvm.return %[[MEMORY]]
