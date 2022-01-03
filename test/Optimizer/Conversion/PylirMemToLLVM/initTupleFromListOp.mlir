// RUN: pylir-opt %s -convert-pylirMem-to-llvm --split-input-file | FileCheck %s


py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.list = #py.type
py.globalValue const @builtins.tuple = #py.type

func @foo(%list : !py.dynamic) -> !py.dynamic {
    %0 = py.constant @builtins.tuple
    %1 = pyMem.gcAllocObject %0
    %2 = pyMem.initTupleFromList %1 to (* %list)
    return %2 : !py.dynamic
}

// CHECK-LABEL: llvm.func @foo
// CHECK-SAME: %[[LIST:[[:alnum:]]+]]
// CHECK-NEXT: %[[TUPLE:.*]] = llvm.mlir.addressof @builtins.tuple
// CHECK-NEXT: %[[TUPLE_CAST:.*]] = llvm.bitcast %[[TUPLE]]
// CHECK-NEXT: %[[NULL:.*]] = llvm.mlir.null
// CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i32)
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[NULL]][%[[ONE]]]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.ptrtoint %[[GEP]]
// CHECK-NEXT: %[[SLOTS:.*]] = llvm.mlir.constant(0 : index)
// CHECK-NEXT: %[[BYTES:.*]] = llvm.add %[[SIZE]], %[[SLOTS]]
// CHECK-NEXT: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc(%[[BYTES]])
// CHECK-NEXT: %[[ZERO_I8:.*]] = llvm.mlir.constant(0 : i8)
// CHECK-NEXT: %[[FALSE:.*]] = llvm.mlir.constant(false)
// CHECK-NEXT: "llvm.intr.memset"(%[[MEMORY]], %[[ZERO_I8]], %[[BYTES]], %[[FALSE]])
// CHECK-NEXT: %[[RESULT:.*]] = llvm.bitcast %[[MEMORY]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[RESULT]][%[[ZERO]], %[[ZERO]]]
// CHECK-NEXT: llvm.store %[[TUPLE_CAST]], %[[GEP]]
// CHECK-NEXT: %[[TUPLE_OBJ:.*]] = llvm.bitcast %[[RESULT]]
// CHECK-NEXT: %[[LIST_OBJ:.*]] = llvm.bitcast %[[LIST]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i32)
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[LIST_OBJ]][%[[ZERO]], %[[ONE]], %[[ZERO]]]
// CHECK-NEXT: %[[LEN:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TUPLE_OBJ]][%[[ZERO]], %[[ONE]], %[[ZERO]]]
// CHECK-NEXT: llvm.store %[[LEN]], %[[GEP]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TUPLE_OBJ]][%[[ZERO]], %[[ONE]], %[[ONE]]]
// CHECK-NEXT: llvm.store %[[LEN]], %[[GEP]]
// CHECK-NEXT: %[[NULL:.*]] = llvm.mlir.null
// CHECK-NEXT: %[[ONE2:.*]] = llvm.mlir.constant(1 : i32)
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[NULL]][%[[ONE2]]]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.ptrtoint %[[GEP]]
// CHECK-NEXT: %[[BYTES:.*]] = llvm.mul %[[LEN]], %[[SIZE]]
// CHECK-NEXT: %[[ARRAY:.*]] = llvm.call @malloc(%[[BYTES]])
// CHECK-NEXT: %[[ARRAY_CAST:.*]] = llvm.bitcast %[[ARRAY]]
// CHECK-NEXT: %[[TWO:.*]] = llvm.mlir.constant(2 : i32)
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TUPLE_OBJ]][%[[ZERO]], %[[ONE]], %[[TWO]]]
// CHECK-NEXT: llvm.store %[[ARRAY_CAST]], %[[GEP]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[LIST_OBJ]][%[[ZERO]], %[[ONE]], %[[TWO]]]
// CHECK-NEXT: %[[LIST_ARRAY:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[TUPLE_ARRAY_i8:.*]] = llvm.bitcast %[[ARRAY_CAST]]
// CHECK-NEXT: %[[LIST_ARRAY_i8:.*]] = llvm.bitcast %[[LIST_ARRAY]]
// CHECK-NEXT: %[[FALSE:.*]] = llvm.mlir.constant(false)
// CHECK-NEXT: "llvm.intr.memcpy"(%[[TUPLE_ARRAY_i8]], %[[LIST_ARRAY_i8]], %[[BYTES]], %[[FALSE]])
// CHECK-NEXT: llvm.return %[[RESULT]]
