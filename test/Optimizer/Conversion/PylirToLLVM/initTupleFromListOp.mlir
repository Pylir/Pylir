// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s


py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.list = #py.type
py.globalValue const @builtins.tuple = #py.type

func.func @foo(%list : !py.dynamic) -> !py.dynamic {
    %0 = py.constant(@builtins.tuple)
    %1 = pyMem.gcAllocObject %0
    %2 = pyMem.initTupleFromList %1 to (* %list)
    return %2 : !py.dynamic
}

// CHECK-LABEL: llvm.func @foo
// CHECK-SAME: %[[LIST:[[:alnum:]]+]]
// CHECK-NEXT: %[[TUPLE:.*]] = llvm.mlir.addressof @builtins.tuple
// CHECK-NEXT: %[[BYTES:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc(%[[BYTES]])
// CHECK-NEXT: %[[ZERO_I8:.*]] = llvm.mlir.constant(0 : i8)
// CHECK-NEXT: %[[FALSE:.*]] = llvm.mlir.constant(false)
// CHECK-NEXT: "llvm.intr.memset"(%[[MEMORY]], %[[ZERO_I8]], %[[BYTES]], %[[FALSE]])
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][%[[ZERO]], 0]
// CHECK-NEXT: llvm.store %[[TUPLE]], %[[GEP]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[SIZE_PTR:.*]] = llvm.getelementptr %[[LIST]][%[[ZERO]], 1]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.load %[[SIZE_PTR]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][%[[ZERO]], 1]
// CHECK-NEXT: llvm.store %[[SIZE]], %[[GEP]]
// CHECK-NEXT: %[[TYPE_ELEMENT_SIZE:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[BYTES:.*]] = llvm.mul %[[SIZE]], %[[TYPE_ELEMENT_SIZE]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[TRAILING:.*]] = llvm.getelementptr %[[MEMORY]][%[[ZERO]], 2]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[ZERO2:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[ARRAY:.*]] = llvm.getelementptr %[[TRAILING]][%[[ZERO]], %[[ZERO2]]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[TUPLE_PTR_PTR:.*]] = llvm.getelementptr %[[LIST]][%[[ZERO]], 2]
// CHECK-NEXT: %[[LOAD:.*]] = llvm.load %[[TUPLE_PTR_PTR]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[TRAILING:.*]] = llvm.getelementptr %[[LOAD]][%[[ZERO]], 2]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[ZERO2:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[PREV_ARRAY:.*]] = llvm.getelementptr %[[TRAILING]][%[[ZERO]], %[[ZERO2]]]
// CHECK-NEXT: %[[FALSE_C:.*]] = llvm.mlir.constant(false)
// CHECK-NEXT: "llvm.intr.memcpy"(%[[ARRAY]], %[[PREV_ARRAY]], %[[BYTES]], %[[FALSE_C]])
// CHECK-NEXT: llvm.return %[[MEMORY]]
