// RUN: pylir-opt %s -convert-pylir-to-llvm --reconcile-unrealized-casts --split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, const, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_list = #py.globalValue<builtins.list, const, initializer = #py.type>
py.external @builtins.list, #builtins_list
#builtins_tuple = #py.globalValue<builtins.tuple, const, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple

py.func @foo(%list : !py.dynamic, %length : index) {
    list_resize %list to %length
    return
}

// CHECK-LABEL: llvm.func @foo
// CHECK-SAME: %[[LIST:[[:alnum:]]+]]
// CHECK-SAME: %[[NEW_LENGTH:[[:alnum:]]+]]
// CHECK-NEXT: %[[TUPLE_PTR_PTR:.*]] = llvm.getelementptr %[[LIST]][0, 2]
// CHECK-NEXT: %[[TUPLE_PTR:.*]] = llvm.load %[[TUPLE_PTR_PTR]]
// CHECK-NEXT: %[[SIZE_PTR:.*]] = llvm.getelementptr %[[LIST]][0, 1]
// CHECK-NEXT: %[[LEN:.*]] = llvm.load %[[SIZE_PTR]]
// CHECK-NEXT: %[[ONE_INDEX:.*]] = llvm.mlir.constant(1 : index)
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TUPLE_PTR]][0, 1]
// CHECK-NEXT: %[[CAPACITY:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[GROW:.*]] = llvm.icmp "ult" %[[CAPACITY]], %[[NEW_LENGTH]]
// CHECK-NEXT: llvm.cond_br %[[GROW]], ^[[GROW_BLOCK:.*]], ^[[END_BLOCK:[[:alnum:]]+]]

// CHECK-NEXT: ^[[GROW_BLOCK]]:
// CHECK-NEXT: %[[SHL:.*]] = llvm.shl %[[CAPACITY]], %[[ONE_INDEX]]
// CHECK-NEXT: %[[NEW_CAP:.*]] = llvm.intr.umax(%[[SHL]], %[[NEW_LENGTH]])
// CHECK-NEXT: %[[TUPLE_TYPE:.*]] = llvm.mlir.addressof @builtins.tuple
// CHECK-NEXT: %[[HEADER_SIZE:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[ELEMENT_SIZE:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[TRAILING_SIZE:.*]] = llvm.mul %[[NEW_CAP]], %[[ELEMENT_SIZE]]
// CHECK-NEXT: %[[BYTES:.*]] = llvm.add %[[TRAILING_SIZE]], %[[HEADER_SIZE]]
// CHECK-NEXT: %[[TUPLE_MEMORY:.*]] = llvm.call @pylir_gc_alloc(%[[BYTES]])
// CHECK-NEXT: %[[ZERO_I8:.*]] = llvm.mlir.constant(0 : i8)
// CHECK-NEXT: "llvm.intr.memset"(%[[TUPLE_MEMORY]], %[[ZERO_I8]], %[[BYTES]])
// CHECK-SAME: isVolatile = false
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TUPLE_MEMORY]][0, 0]
// CHECK-NEXT: llvm.store %[[TUPLE_TYPE]], %[[GEP]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TUPLE_MEMORY]][0, 1]
// CHECK-NEXT: llvm.store %[[NEW_CAP]], %[[GEP]]
// CHECK-NEXT: %[[TRAILING:.*]] = llvm.getelementptr %[[TUPLE_MEMORY]][0, 2]
// CHECK-NEXT: %[[ARRAY:.*]] = llvm.getelementptr %[[TRAILING]][0, 0]
// CHECK-NEXT: %[[TRAILING:.*]] = llvm.getelementptr %[[TUPLE_PTR]][0, 2]
// CHECK-NEXT: %[[PREV_ARRAY:.*]] = llvm.getelementptr %[[TRAILING]][0, 0]
// CHECK-NEXT: %[[ELEMENT_SIZE:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[TRAILING_SIZE:.*]] = llvm.mul %[[LEN]], %[[ELEMENT_SIZE]]
// CHECK-NEXT: "llvm.intr.memcpy"(%[[ARRAY]], %[[PREV_ARRAY]], %[[TRAILING_SIZE]])
// CHECK-SAME: isVolatile = false
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[LIST]][0, 2]
// CHECK-NEXT: llvm.store %[[TUPLE_MEMORY]], %[[GEP]]
// CHECK-NEXT: llvm.br ^[[END_BLOCK]]

// CHECK-NEXT: ^[[END_BLOCK]]:
// CHECK-NEXT: llvm.store %[[NEW_LENGTH]], %[[SIZE_PTR]]
// CHECK-NEXT: llvm.return
