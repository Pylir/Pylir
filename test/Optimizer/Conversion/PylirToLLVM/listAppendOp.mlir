// RUN: pylir-opt %s -convert-pylir-to-llvm --reconcile-unrealized-casts --split-input-file | FileCheck %s

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
// CHECK-NEXT: %[[TUPLE_PTR_PTR:.*]] = llvm.getelementptr %[[LIST_CAST]][0, 2]
// CHECK-NEXT: %[[TUPLE_PTR:.*]] = llvm.load %[[TUPLE_PTR_PTR]]
// CHECK-NEXT: %[[SIZE_PTR:.*]] = llvm.getelementptr %[[LIST_CAST]][0, 1]
// CHECK-NEXT: %[[LEN:.*]] = llvm.load %[[SIZE_PTR]]
// CHECK-NEXT: %[[ONE_INDEX:.*]] = llvm.mlir.constant(1 : index)
// CHECK-NEXT: %[[INCREMENTED:.*]] = llvm.add %[[LEN]], %[[ONE_INDEX]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TUPLE_PTR]][0, 1]
// CHECK-NEXT: %[[CAPACITY:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[GROW:.*]] = llvm.icmp "ult" %[[CAPACITY]], %[[INCREMENTED]]
// CHECK-NEXT: llvm.cond_br %[[GROW]], ^[[GROW_BLOCK:.*]], ^[[DONT_GROW_BLOCK:[[:alnum:]]+]]

// CHECK-NEXT: ^[[DONT_GROW_BLOCK]]:
// CHECK-NEXT: %[[TRAILING:.*]] = llvm.getelementptr %[[TUPLE_PTR]][0, 2]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TRAILING]][0, %[[LEN]]]
// CHECK-NEXT: llvm.store %[[ITEM]], %[[GEP]]
// CHECK-NEXT: llvm.br ^[[END_BLOCK:[[:alnum:]]+]]

// CHECK-NEXT: ^[[GROW_BLOCK]]:
// CHECK-NEXT: %[[SHL:.*]] = llvm.shl %[[CAPACITY]], %[[ONE_INDEX]]
// CHECK-NEXT: %[[NEW_CAP:.*]] = "llvm.intr.umax"(%[[SHL]], %[[INCREMENTED]])
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TUPLE_PTR]][0, 0]
// CHECK-NEXT: %[[TUPLE_TYPE:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[ELEMENT_SIZE:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[TRAILING_SIZE:.*]] = llvm.mul %[[NEW_CAP]], %[[ELEMENT_SIZE]]
// CHECK-NEXT: %[[HEADER_SIZE:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[BYTES:.*]] = llvm.add %[[TRAILING_SIZE]], %[[HEADER_SIZE]]
// CHECK-NEXT: %[[TUPLE_MEMORY:.*]] = llvm.call @pylir_gc_alloc(%[[BYTES]])
// CHECK-NEXT: %[[ZERO_I8:.*]] = llvm.mlir.constant(0 : i8)
// CHECK-NEXT: %[[FALSE_C:.*]] = llvm.mlir.constant(false)
// CHECK-NEXT: "llvm.intr.memset"(%[[TUPLE_MEMORY]], %[[ZERO_I8]], %[[BYTES]], %[[FALSE_C]])
// CHECK-NEXT: %[[TUPLE_RESULT:.*]] = llvm.bitcast %[[TUPLE_MEMORY]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TUPLE_RESULT]][0, 0]
// CHECK-NEXT: llvm.store %[[TUPLE_TYPE]], %[[GEP]]
// CHECK-NEXT: %[[TUPLE:.*]] = llvm.bitcast %[[TUPLE_RESULT]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TUPLE]][0, 1]
// CHECK-NEXT: llvm.store %[[NEW_CAP]], %[[GEP]]
// CHECK-NEXT: %[[TRAILING:.*]] = llvm.getelementptr %[[TUPLE]][0, 2]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TRAILING]][0, %[[LEN]]]
// CHECK-NEXT: llvm.store %[[ITEM]], %[[GEP]]
// CHECK-NEXT: %[[ARRAY:.*]] = llvm.getelementptr %[[TRAILING]][0, 0]
// CHECK-NEXT: %[[TRAILING:.*]] = llvm.getelementptr %[[TUPLE_PTR]][0, 2]
// CHECK-NEXT: %[[PREV_ARRAY:.*]] = llvm.getelementptr %[[TRAILING]][0, 0]
// CHECK-NEXT: %[[ARRAY_I8:.*]] = llvm.bitcast %[[ARRAY]]
// CHECK-NEXT: %[[PREV_ARRAY_I8:.*]] = llvm.bitcast %[[PREV_ARRAY]]
// CHECK-NEXT: %[[ELEMENT_SIZE:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[TRAILING_SIZE:.*]] = llvm.mul %[[LEN]], %[[ELEMENT_SIZE]]
// CHECK-NEXT: %[[FALSE_C:.*]] = llvm.mlir.constant(false)
// CHECK-NEXT: "llvm.intr.memcpy"(%[[ARRAY_I8]], %[[PREV_ARRAY_I8]], %[[TRAILING_SIZE]], %[[FALSE_C]])
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[LIST_CAST]][0, 2]
// CHECK-NEXT: llvm.store %[[TUPLE]], %[[GEP]]
// CHECK-NEXT: llvm.br ^[[END_BLOCK]]

// CHECK-NEXT: ^[[END_BLOCK]]:
// CHECK-NEXT: llvm.store %[[INCREMENTED]], %[[SIZE_PTR]]
// CHECK-NEXT: llvm.return
