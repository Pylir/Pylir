// RUN: pylir-opt %s -convert-pylirMem-to-llvm --split-input-file | FileCheck %s


py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.dict = #py.type

func @foo() -> !py.dynamic {
    %0 = py.constant @builtins.dict
    %1 = pyMem.gcAllocObject %0
    %2 = pyMem.initDict %1
    return %2 : !py.dynamic
}

// CHECK-LABEL: llvm.func @foo
// CHECK-NEXT: %[[DICT:.*]] = llvm.mlir.addressof @builtins.dict
// CHECK-NEXT: %[[DICT_CAST:.*]] = llvm.bitcast %[[DICT]]
// CHECK-NEXT: %[[NULL:.*]] = llvm.mlir.null
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[NULL]][1]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.ptrtoint %[[GEP]]
// CHECK-NEXT: %[[SLOTS:.*]] = llvm.mlir.constant(0 : index)
// CHECK-NEXT: %[[BYTES:.*]] = llvm.add %[[SIZE]], %[[SLOTS]]
// CHECK-NEXT: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc(%[[BYTES]])
// CHECK-NEXT: %[[ZERO_I8:.*]] = llvm.mlir.constant(0 : i8)
// CHECK-NEXT: %[[FALSE:.*]] = llvm.mlir.constant(false)
// CHECK-NEXT: "llvm.intr.memset"(%[[MEMORY]], %[[ZERO_I8]], %[[BYTES]], %[[FALSE]])
// CHECK-NEXT: %[[RESULT:.*]] = llvm.bitcast %[[MEMORY]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[RESULT]][0, 0]
// CHECK-NEXT: llvm.store %[[DICT_CAST]], %[[GEP]]
// CHECK-NEXT: llvm.return %[[RESULT]]
