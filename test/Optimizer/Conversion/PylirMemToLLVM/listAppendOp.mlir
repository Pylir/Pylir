// RUN: pylir-opt %s -convert-pylirMem-to-llvm --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.list = #py.type

func @foo(%list : !py.dynamic, %item : !py.dynamic) {
    py.list.append %list, %item
    return
}

// CHECK-LABEL: llvm.func @foo
// CHECK-SAME: %[[LIST:[[:alnum:]]+]]
// CHECK-SAME: %[[ITEM:[[:alnum:]]+]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i32)
// CHECK-NEXT: %[[LIST_CAST:.*]] = llvm.bitcast %[[LIST]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[LIST_CAST]][%[[ZERO]], %[[ONE]], %[[ZERO]]]
// CHECK-NEXT: %[[LEN:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[ONE_INDEX:.*]] = llvm.mlir.constant(1 : index)
// CHECK-NEXT: %[[INCREMENTED:.*]] = llvm.add %[[LEN]], %[[ONE_INDEX]]
// CHECK-NEXT: llvm.store %[[INCREMENTED:.*]], %[[GEP]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[LIST_CAST]][%[[ZERO]], %[[ONE]], %[[ONE]]]
// CHECK-NEXT: %[[CAPACITY:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[GROW:.*]] = llvm.icmp "ult" %[[CAPACITY]], %[[INCREMENTED]]
// CHECK-NEXT: llvm.cond_br %[[GROW]], ^[[GROW_BLOCK:.*]], ^[[END_BLOCK:[[:alnum:]]+]]
// CHECK-NEXT: ^[[GROW_BLOCK]]:
// CHECK-NEXT: %[[TWO:.*]] = llvm.mlir.constant(2 : i32)
// CHECK-NEXT: %[[SHL:.*]] = llvm.shl %[[CAPACITY]], %[[ONE_INDEX]]
// CHECK-NEXT: %[[NEW_CAP:.*]] = "llvm.intr.umin"(%[[SHL]], %[[INCREMENTED]])
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[LIST_CAST]][%[[ZERO]], %[[ONE]], %[[TWO]]]
// CHECK-NEXT: %[[ARRAY:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[NULL:.*]] = llvm.mlir.null
// CHECK-NEXT: %[[ONE2:.*]] = llvm.mlir.constant(1 : i32)
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[NULL]][%[[ONE2]]]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.ptrtoint %[[GEP2]]
// CHECK-NEXT: %[[BYTES:.*]] = llvm.mul %[[NEW_CAP]], %[[SIZE]]
// CHECK-NEXT: %[[NEW_MEMORY:.*]] = llvm.call @pylir_gc_alloc(%[[BYTES]])
// CHECK-NEXT: %[[NEW_ARRAY:.*]] = llvm.bitcast %[[NEW_MEMORY]]
// CHECK-NEXT: %[[ARRAY_I8:.*]] = llvm.bitcast %[[ARRAY]]
// CHECK-NEXT: %[[NEW_ARRAY_I8:.*]] = llvm.bitcast %[[NEW_ARRAY]]
// CHECK-NEXT: %[[FALSE:.*]] = llvm.mlir.constant(false)
// CHECK-NEXT: %[[BYTES:.*]] = llvm.mul %[[LEN]], %[[SIZE]]
// CHECK-NEXT: "llvm.intr.memcpy"(%[[NEW_ARRAY_I8]], %[[ARRAY_I8]], %[[BYTES]], %[[FALSE]])
// CHECK-NEXT: llvm.store %[[NEW_ARRAY]], %[[GEP]]
// CHECK-NEXT: llvm.br ^[[END_BLOCK]]
// CHECK-NEXT: ^[[END_BLOCK]]:
// CHECK-NEXT: %[[TWO:.*]] = llvm.mlir.constant(2 : i32)
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[LIST_CAST]][%[[ZERO]], %[[ONE]], %[[TWO]]]
// CHECK-NEXT: %[[ARRAY:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[ARRAY]][%[[LEN]]]
// CHECK-NEXT: llvm.store %[[ITEM]], %[[GEP]]
// CHECK-NEXT: llvm.return
