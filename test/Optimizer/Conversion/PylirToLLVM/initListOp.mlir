// RUN: pylir-opt %s -convert-pylir-to-llvm --reconcile-unrealized-casts --split-input-file | FileCheck %s


py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.list = #py.type

func @foo() -> !py.dynamic {
    %0 = py.constant @builtins.list
    %1 = pyMem.gcAllocObject %0
    %2 = pyMem.initList %1 to [%0]
    return %2 : !py.dynamic
}

// CHECK-LABEL: llvm.func @foo
// CHECK-NEXT: %[[LIST:.*]] = llvm.mlir.addressof @builtins.list
// CHECK-NEXT: %[[LIST_CAST:.*]] = llvm.bitcast %[[LIST]]
// CHECK-NEXT: %[[BYTES:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc(%[[BYTES]])
// CHECK-NEXT: %[[ZERO_I8:.*]] = llvm.mlir.constant(0 : i8)
// CHECK-NEXT: %[[FALSE:.*]] = llvm.mlir.constant(false)
// CHECK-NEXT: "llvm.intr.memset"(%[[MEMORY]], %[[ZERO_I8]], %[[BYTES]], %[[FALSE]])
// CHECK-NEXT: %[[RESULT:.*]] = llvm.bitcast %[[MEMORY]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[RESULT]][0, 0]
// CHECK-NEXT: llvm.store %[[LIST_CAST]], %[[GEP]]
// CHECK-NEXT: %[[LIST_OBJ:.*]] = llvm.bitcast %[[RESULT]]
// CHECK-NEXT: %[[LEN:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[SIZE_PTR:.*]] = llvm.getelementptr %[[LIST_OBJ]][0, 1]
// CHECK-NEXT: llvm.store %[[LEN]], %[[SIZE_PTR]]
// CHECK-NEXT: %[[TUPLE_TYPE:.*]] = llvm.mlir.addressof @builtins.tuple
// CHECK-NEXT: %[[CASTED:.*]] = llvm.bitcast %[[TUPLE_TYPE]]
// CHECK-NEXT: %[[ELEMENT_SIZE:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[TRAILING_SIZE:.*]] = llvm.mul %[[LEN]], %[[ELEMENT_SIZE]]
// CHECK-NEXT: %[[HEADER_SIZE:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[BYTES:.*]] = llvm.add %[[TRAILING_SIZE]], %[[HEADER_SIZE]]
// CHECK-NEXT: %[[TUPLE_MEMORY:.*]] = llvm.call @pylir_gc_alloc(%[[BYTES]])
// CHECK-NEXT: %[[ZERO_I8:.*]] = llvm.mlir.constant(0 : i8)
// CHECK-NEXT: %[[FALSE:.*]] = llvm.mlir.constant(false)
// CHECK-NEXT: "llvm.intr.memset"(%[[TUPLE_MEMORY]], %[[ZERO_I8]], %[[BYTES]], %[[FALSE]])
// CHECK-NEXT: %[[TUPLE_RESULT:.*]] = llvm.bitcast %[[TUPLE_MEMORY]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TUPLE_RESULT]][0, 0]
// CHECK-NEXT: llvm.store %[[CASTED]], %[[GEP]]
// CHECK-NEXT: %[[TUPLE:.*]] = llvm.bitcast %[[TUPLE_RESULT]]
// CHECK-NEXT: %[[CAPACITY:.*]] = llvm.mlir.constant(1 : i{{.*}})
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TUPLE]][0, 1]
// CHECK-NEXT: llvm.store %[[CAPACITY]], %[[GEP]]
// CHECK-NEXT: %[[TRAILING:.*]] = llvm.getelementptr %[[TUPLE]][0, 2]
// CHECK-NEXT: %[[FIRST:.*]] = llvm.getelementptr %[[TRAILING]][0, 0]
// CHECK-NEXT: llvm.store %[[LIST_CAST]], %[[FIRST]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[LIST_OBJ]][0, 2]
// CHECK-NEXT: %[[CASTED:.*]] = llvm.bitcast %[[TUPLE_RESULT]]
// CHECK-NEXT: llvm.store %[[CASTED]], %[[GEP]]
// CHECK-NEXT: llvm.return %[[RESULT]]
