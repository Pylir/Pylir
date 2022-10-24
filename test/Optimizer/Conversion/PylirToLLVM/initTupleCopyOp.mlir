// RUN: pylir-opt %s -convert-arith-to-llvm -convert-pylir-to-llvm --reconcile-unrealized-casts --split-input-file | FileCheck %s


py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.list = #py.type
py.globalValue const @builtins.tuple = #py.type

func.func @foo(%tuple : !py.dynamic) -> !py.dynamic {
    %0 = py.constant(#py.ref<@builtins.tuple>)
    %c0 = arith.constant 0 : index
    %1 = pyMem.gcAllocObject %0[%c0]
    %2 = pyMem.initTupleCopy %1 to (* %tuple)
    return %2 : !py.dynamic
}

// CHECK-LABEL: llvm.func @foo
// CHECK-SAME: %[[SOURCE:[[:alnum:]]+]]
// CHECK-NEXT: %[[TUPLE:.*]] = llvm.mlir.addressof @builtins.tuple
// CHECK: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc
// CHECK: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][0, 0]
// CHECK-NEXT: llvm.store %[[TUPLE]], %[[GEP]]
// CHECK-NEXT: %[[SIZE_PTR:.*]] = llvm.getelementptr %[[SOURCE]][0, 1]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.load %[[SIZE_PTR]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][0, 1]
// CHECK-NEXT: llvm.store %[[SIZE]], %[[GEP]]
// CHECK-NEXT: %[[TYPE_ELEMENT_SIZE:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[BYTES:.*]] = llvm.mul %[[SIZE]], %[[TYPE_ELEMENT_SIZE]]
// CHECK-NEXT: %[[TRAILING:.*]] = llvm.getelementptr %[[MEMORY]][0, 2]
// CHECK-NEXT: %[[ARRAY:.*]] = llvm.getelementptr %[[TRAILING]][0, 0]
// CHECK-NEXT: %[[TRAILING:.*]] = llvm.getelementptr %[[SOURCE]][0, 2]
// CHECK-NEXT: %[[PREV_ARRAY:.*]] = llvm.getelementptr %[[TRAILING]][0, 0]
// CHECK-NEXT: %[[FALSE_C:.*]] = llvm.mlir.constant(false)
// CHECK-NEXT: "llvm.intr.memcpy"(%[[ARRAY]], %[[PREV_ARRAY]], %[[BYTES]], %[[FALSE_C]])
// CHECK-NEXT: llvm.return %[[MEMORY]]
