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
// CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i32)
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[NULL]][%[[ONE]]]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.ptrtoint %[[GEP]]
// CHECK-NEXT: %[[SLOTS:.*]] = llvm.mlir.constant(0 : index)
// CHECK-NEXT: %[[BYTES:.*]] = llvm.add %[[SIZE]], %[[SLOTS]]
// CHECK-NEXT: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc(%[[BYTES]])
// CHECK-NEXT: %[[RESULT:.*]] = llvm.bitcast %[[MEMORY]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[RESULT]][%[[ZERO]], %[[ZERO]]]
// CHECK-NEXT: llvm.store %[[DICT_CAST]], %[[GEP]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i32)
// CHECK-NEXT: %[[TWO:.*]] = llvm.mlir.constant(2 : i32)
// CHECK-NEXT: %[[THREE:.*]] = llvm.mlir.constant(3 : i32)
// CHECK-NEXT: %[[DICT_OBJ:.*]] = llvm.bitcast %[[RESULT]]
// CHECK-NEXT: %[[ZERO_I:.*]] = llvm.mlir.constant(0 : index)
// CHECK-NEXT: %[[NULL_PAIR:.*]] = llvm.mlir.null
// CHECK-NEXT: %[[NULL_BUCKET:.*]] = llvm.mlir.null
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[DICT_OBJ]][%[[ZERO]], %[[ONE]], %[[ZERO]]]
// CHECK-NEXT: llvm.store %[[ZERO_I]], %[[GEP]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[DICT_OBJ]][%[[ZERO]], %[[ONE]], %[[ONE]]]
// CHECK-NEXT: llvm.store %[[ZERO_I]], %[[GEP]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[DICT_OBJ]][%[[ZERO]], %[[ONE]], %[[TWO]]]
// CHECK-NEXT: llvm.store %[[NULL_PAIR]], %[[GEP]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[DICT_OBJ]][%[[ZERO]], %[[TWO]]]
// CHECK-NEXT: llvm.store %[[ZERO_I]], %[[GEP]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[DICT_OBJ]][%[[ZERO]], %[[THREE]]]
// CHECK-NEXT: llvm.store %[[NULL_BUCKET]], %[[GEP]]
// CHECK-NEXT: llvm.return %[[RESULT]]
