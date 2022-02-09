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
// CHECK-NEXT: %[[BYTES:.*]] = llvm.mlir.constant(32 : index)
// CHECK-NEXT: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc(%[[BYTES]])
// CHECK-NEXT: %[[ZERO_I8:.*]] = llvm.mlir.constant(0 : i8)
// CHECK-NEXT: %[[FALSE:.*]] = llvm.mlir.constant(false)
// CHECK-NEXT: "llvm.intr.memset"(%[[MEMORY]], %[[ZERO_I8]], %[[BYTES]], %[[FALSE]])
// CHECK-NEXT: %[[RESULT:.*]] = llvm.bitcast %[[MEMORY]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[RESULT]][0, 0]
// CHECK-NEXT: llvm.store %[[LIST_CAST]], %[[GEP]]
// CHECK-NEXT: %[[LIST_OBJ:.*]] = llvm.bitcast %[[RESULT]]
// CHECK-NEXT: %[[BUFFER:.*]] = llvm.getelementptr %[[LIST_OBJ]][0, 1]
// CHECK-NEXT: %[[LEN:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[BUFFER]][0, 0]
// CHECK-NEXT: llvm.store %[[LEN]], %[[GEP]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[BUFFER]][0, 1]
// CHECK-NEXT: llvm.store %[[LEN]], %[[GEP]]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.mlir.constant(8 : index)
// CHECK-NEXT: %[[BYTES:.*]] = llvm.mul %[[LEN]], %[[SIZE]]
// CHECK-NEXT: %[[ARRAY:.*]] = llvm.call @malloc(%[[BYTES]])
// CHECK-NEXT: %[[ARRAY_CAST:.*]] = llvm.bitcast %[[ARRAY]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[BUFFER]][0, 2]
// CHECK-NEXT: llvm.store %[[ARRAY_CAST]], %[[GEP]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[ARRAY_CAST]][0]
// CHECK-NEXT: llvm.store %[[LIST_CAST]], %[[GEP]]
// CHECK-NEXT: llvm.return %[[RESULT]]
