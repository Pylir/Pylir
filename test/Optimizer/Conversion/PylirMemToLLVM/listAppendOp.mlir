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
// CHECK-NEXT: %[[LIST_CAST:.*]] = llvm.bitcast %[[LIST]]
// CHECK-NEXT: %[[BUFFER:.*]] = llvm.getelementptr %[[LIST_CAST]][0, 1]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[BUFFER]][0, 0]
// CHECK-NEXT: %[[LEN:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[ONE_INDEX:.*]] = llvm.mlir.constant(1 : index)
// CHECK-NEXT: %[[INCREMENTED:.*]] = llvm.add %[[LEN]], %[[ONE_INDEX]]
// CHECK-NEXT: llvm.store %[[INCREMENTED:.*]], %[[GEP]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[BUFFER]][0, 1]
// CHECK-NEXT: %[[CAPACITY:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[GROW:.*]] = llvm.icmp "ult" %[[CAPACITY]], %[[INCREMENTED]]
// CHECK-NEXT: llvm.cond_br %[[GROW]], ^[[GROW_BLOCK:.*]], ^[[END_BLOCK:[[:alnum:]]+]]
// CHECK-NEXT: ^[[GROW_BLOCK]]:
// CHECK-NEXT: %[[SHL:.*]] = llvm.shl %[[CAPACITY]], %[[ONE_INDEX]]
// CHECK-NEXT: %[[NEW_CAP:.*]] = "llvm.intr.umax"(%[[SHL]], %[[INCREMENTED]])
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[BUFFER]][0, 2]
// CHECK-NEXT: %[[ARRAY:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.mlir.constant(8 : index)
// CHECK-NEXT: %[[BYTES:.*]] = llvm.mul %[[NEW_CAP]], %[[SIZE]]
// CHECK-NEXT: %[[ARRAY_I8:.*]] = llvm.bitcast %[[ARRAY]]
// CHECK-NEXT: %[[NEW_MEMORY:.*]] = llvm.call @realloc(%[[ARRAY_I8]], %[[BYTES]])
// CHECK-NEXT: %[[NEW_ARRAY:.*]] = llvm.bitcast %[[NEW_MEMORY]]
// CHECK-NEXT: llvm.store %[[NEW_ARRAY]], %[[GEP]]
// CHECK-NEXT: llvm.br ^[[END_BLOCK]]
// CHECK-NEXT: ^[[END_BLOCK]]:
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[BUFFER]][0, 2]
// CHECK-NEXT: %[[ARRAY:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[ARRAY]][%[[LEN]]]
// CHECK-NEXT: llvm.store %[[ITEM]], %[[GEP]]
// CHECK-NEXT: llvm.return
