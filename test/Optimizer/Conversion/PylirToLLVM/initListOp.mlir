// RUN: pylir-opt %s -convert-arith-to-llvm -convert-pylir-to-llvm --reconcile-unrealized-casts --split-input-file | FileCheck %s


py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.list = #py.type

py.func @foo() -> !py.dynamic {
    %0 = constant(#py.ref<@builtins.list>)
    %c0 = arith.constant 0 : index
    %1 = pyMem.gcAllocObject %0[%c0]
    %2 = pyMem.initList %1 to [%0]
    return %2 : !py.dynamic
}

// CHECK-LABEL: llvm.func @foo
// CHECK-NEXT: %[[LIST:.*]] = llvm.mlir.addressof @builtins.list
// CHECK: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc
// CHECK-NEXT: %[[ZERO_I8:.*]] = llvm.mlir.constant(0 : i8)
// CHECK: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][0, 0]
// CHECK-NEXT: llvm.store %[[LIST]], %[[GEP]]
// CHECK-NEXT: %[[LEN:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[SIZE_PTR:.*]] = llvm.getelementptr %[[MEMORY]][0, 1]
// CHECK-NEXT: llvm.store %[[LEN]], %[[SIZE_PTR]] {tbaa = [@tbaa::@"Python List Size access"]}
// CHECK-NEXT: %[[TUPLE_TYPE:.*]] = llvm.mlir.addressof @builtins.tuple
// CHECK-NEXT: %[[HEADER_SIZE:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[ELEMENT_SIZE:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[TRAILING_SIZE:.*]] = llvm.mul %[[LEN]], %[[ELEMENT_SIZE]]
// CHECK-NEXT: %[[BYTES:.*]] = llvm.add %[[TRAILING_SIZE]], %[[HEADER_SIZE]]
// CHECK-NEXT: %[[TUPLE_MEMORY:.*]] = llvm.call @pylir_gc_alloc(%[[BYTES]])
// CHECK-NEXT: %[[ZERO_I8:.*]] = llvm.mlir.constant(0 : i8)
// CHECK-NEXT: %[[FALSE:.*]] = llvm.mlir.constant(false)
// CHECK-NEXT: "llvm.intr.memset"(%[[TUPLE_MEMORY]], %[[ZERO_I8]], %[[BYTES]], %[[FALSE]])
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TUPLE_MEMORY]][0, 0]
// CHECK-NEXT: llvm.store %[[TUPLE_TYPE]], %[[GEP]]
// CHECK-NEXT: %[[CAPACITY:.*]] = llvm.mlir.constant(1 : i{{.*}})
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TUPLE_MEMORY]][0, 1]
// CHECK-NEXT: llvm.store %[[CAPACITY]], %[[GEP]] {tbaa = [@tbaa::@"Python Tuple Size access"]}
// CHECK-NEXT: %[[TRAILING:.*]] = llvm.getelementptr %[[TUPLE_MEMORY]][0, 2]
// CHECK-NEXT: %[[FIRST:.*]] = llvm.getelementptr %[[TRAILING]][0, 0]
// CHECK-NEXT: llvm.store %[[LIST]], %[[FIRST]] {tbaa = [@tbaa::@"Python Tuple Elements access"]}
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][0, 2]
// CHECK-NEXT: llvm.store %[[TUPLE_MEMORY]], %[[GEP]] {tbaa = [@tbaa::@"Python List Tuple access"]}
// CHECK-NEXT: llvm.return %[[MEMORY]]
