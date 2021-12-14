// RUN: pylir-opt %s -convert-pylirMem-to-llvm --split-input-file | FileCheck %s


py.globalValue const @builtins.type = #py.type
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
// CHECK-NEXT: %[[NULL:.*]] = llvm.mlir.null
// CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i32)
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[NULL]][%[[ONE]]]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.ptrtoint %[[GEP]]
// CHECK-NEXT: %[[SLOTS:.*]] = llvm.mlir.constant(0 : index)
// CHECK-NEXT: %[[BYTES:.*]] = llvm.add %[[SIZE]], %[[SLOTS]]
// CHECK-NEXT: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc(%[[BYTES]])
// CHECK-NEXT: %[[RESULT:.*]] = llvm.bitcast %[[MEMORY]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[RESULT]][%[[ZERO]], %[[ZERO]]]
// CHECK-NEXT: llvm.store %[[LIST_CAST]], %[[GEP]]
// CHECK-NEXT: %[[LIST_OBJ:.*]] = llvm.bitcast %[[RESULT]]
// CHECK-NEXT: %[[LEN:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i32)
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[LIST_OBJ]][%[[ZERO]], %[[ONE]], %[[ZERO]]]
// CHECK-NEXT: llvm.store %[[LEN]], %[[GEP]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[LIST_OBJ]][%[[ZERO]], %[[ONE]], %[[ONE]]]
// CHECK-NEXT: llvm.store %[[LEN]], %[[GEP]]
// CHECK-NEXT: %[[NULL:.*]] = llvm.mlir.null
// CHECK-NEXT: %[[ONE2:.*]] = llvm.mlir.constant(1 : i32)
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[NULL]][%[[ONE2]]]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.ptrtoint %[[GEP]]
// CHECK-NEXT: %[[BYTES:.*]] = llvm.mul %[[LEN]], %[[SIZE]]
// CHECK-NEXT: %[[ARRAY:.*]] = llvm.call @pylir_gc_alloc(%[[BYTES]])
// CHECK-NEXT: %[[ARRAY_CAST:.*]] = llvm.bitcast %[[ARRAY]]
// CHECK-NEXT: %[[TWO:.*]] = llvm.mlir.constant(2 : i32)
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[LIST_OBJ]][%[[ZERO]], %[[ONE]], %[[TWO]]]
// CHECK-NEXT: llvm.store %[[ARRAY_CAST]], %[[GEP]]
// CHECK-NEXT: %[[OFFSET:.*]] = llvm.mlir.constant(0 : i32)
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[ARRAY_CAST]][%[[OFFSET]]]
// CHECK-NEXT: llvm.store %[[LIST_CAST]], %[[GEP]]
// CHECK-NEXT: llvm.return %[[RESULT]]
