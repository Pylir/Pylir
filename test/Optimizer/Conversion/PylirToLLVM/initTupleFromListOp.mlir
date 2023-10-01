// RUN: pylir-opt %s -convert-arith-to-llvm -convert-pylir-to-llvm --reconcile-unrealized-casts --split-input-file | FileCheck %s


#builtins_type = #py.globalValue<builtins.type, const, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_list = #py.globalValue<builtins.list, const, initializer = #py.type>
py.external @builtins.list, #builtins_list
#builtins_tuple = #py.globalValue<builtins.tuple, const, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple

py.func @foo(%list : !py.dynamic) -> !py.dynamic {
    %0 = constant(#builtins_tuple)
    %c0 = arith.constant 0 : index
    %1 = pyMem.gcAllocObject %0[%c0]
    %2 = pyMem.initTupleFromList %1 to (* %list)
    return %2 : !py.dynamic
}

// CHECK-LABEL: llvm.func @foo
// CHECK-SAME: %[[LIST:[[:alnum:]]+]]
// CHECK-NEXT: %[[TUPLE:.*]] = llvm.mlir.addressof @builtins.tuple
// CHECK: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc
// CHECK: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][0, 0]
// CHECK-NEXT: llvm.store %[[TUPLE]], %[[GEP]]
// CHECK-NEXT: %[[SIZE_PTR:.*]] = llvm.getelementptr %[[LIST]][0, 1]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.load %[[SIZE_PTR]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][0, 1]
// CHECK-NEXT: llvm.store %[[SIZE]], %[[GEP]]
// CHECK-NEXT: %[[TYPE_ELEMENT_SIZE:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[BYTES:.*]] = llvm.mul %[[SIZE]], %[[TYPE_ELEMENT_SIZE]]
// CHECK-NEXT: %[[TRAILING:.*]] = llvm.getelementptr %[[MEMORY]][0, 2]
// CHECK-NEXT: %[[ARRAY:.*]] = llvm.getelementptr %[[TRAILING]][0, 0]
// CHECK-NEXT: %[[TUPLE_PTR_PTR:.*]] = llvm.getelementptr %[[LIST]][0, 2]
// CHECK-NEXT: %[[LOAD:.*]] = llvm.load %[[TUPLE_PTR_PTR]]
// CHECK-NEXT: %[[TRAILING:.*]] = llvm.getelementptr %[[LOAD]][0, 2]
// CHECK-NEXT: %[[PREV_ARRAY:.*]] = llvm.getelementptr %[[TRAILING]][0, 0]
// CHECK-NEXT: "llvm.intr.memcpy"(%[[ARRAY]], %[[PREV_ARRAY]], %[[BYTES]])
// CHECK-SAME: isVolatile = false
// CHECK-NEXT: llvm.return %[[MEMORY]]
