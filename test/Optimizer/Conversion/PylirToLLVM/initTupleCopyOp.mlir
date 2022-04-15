// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s


py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.list = #py.type
py.globalValue const @builtins.tuple = #py.type

func @foo(%tuple : !py.dynamic) -> !py.dynamic {
    %0 = py.constant(@builtins.tuple)
    %1 = pyMem.gcAllocObject %0
    %2 = pyMem.initTupleCopy %1 to (* %tuple)
    return %2 : !py.dynamic
}

// CHECK-LABEL: llvm.func @foo
// CHECK-SAME: %[[SOURCE:[[:alnum:]]+]]
// CHECK-NEXT: %[[TUPLE:.*]] = llvm.mlir.addressof @builtins.tuple
// CHECK-NEXT: %[[TUPLE_CAST:.*]] = llvm.bitcast %[[TUPLE]]
// CHECK-NEXT: %[[BYTES:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc(%[[BYTES]])
// CHECK-NEXT: %[[ZERO_I8:.*]] = llvm.mlir.constant(0 : i8)
// CHECK-NEXT: %[[FALSE:.*]] = llvm.mlir.constant(false)
// CHECK-NEXT: "llvm.intr.memset"(%[[MEMORY]], %[[ZERO_I8]], %[[BYTES]], %[[FALSE]])
// CHECK-NEXT: %[[RESULT:.*]] = llvm.bitcast %[[MEMORY]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[RESULT]][0, 0]
// CHECK-NEXT: llvm.store %[[TUPLE_CAST]], %[[GEP]]
// CHECK-NEXT: %[[TUPLE_OBJ:.*]] = llvm.bitcast %[[RESULT]]
// CHECK-NEXT: %[[SOURCE_CAST:.*]] = llvm.bitcast %[[SOURCE]]
// CHECK-NEXT: %[[SIZE_PTR:.*]] = llvm.getelementptr %[[SOURCE_CAST]][0, 1]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.load %[[SIZE_PTR]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TUPLE_OBJ]][0, 1]
// CHECK-NEXT: llvm.store %[[SIZE]], %[[GEP]]
// CHECK-NEXT: %[[TYPE_ELEMENT_SIZE:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[BYTES:.*]] = llvm.mul %[[SIZE]], %[[TYPE_ELEMENT_SIZE]]
// CHECK-NEXT: %[[TRAILING:.*]] = llvm.getelementptr %[[TUPLE_OBJ]][0, 2]
// CHECK-NEXT: %[[ARRAY:.*]] = llvm.getelementptr %[[TRAILING]][0, 0]
// CHECK-NEXT: %[[TRAILING:.*]] = llvm.getelementptr %[[SOURCE_CAST]][0, 2]
// CHECK-NEXT: %[[PREV_ARRAY:.*]] = llvm.getelementptr %[[TRAILING]][0, 0]
// CHECK-NEXT: %[[ARRAY_I8:.*]] = llvm.bitcast %[[ARRAY]]
// CHECK-NEXT: %[[PREV_ARRAY_I8:.*]] = llvm.bitcast %[[PREV_ARRAY]]
// CHECK-NEXT: %[[FALSE_C:.*]] = llvm.mlir.constant(false)
// CHECK-NEXT: "llvm.intr.memcpy"(%[[ARRAY_I8]], %[[PREV_ARRAY_I8]], %[[BYTES]], %[[FALSE_C]])
// CHECK-NEXT: llvm.return %[[RESULT]]
