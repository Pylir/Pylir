// RUN: pylir-opt %s -convert-pylirMem-to-llvm --split-input-file | FileCheck %s


py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.str = #py.type

func @foo(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = py.constant @builtins.str
    %1 = pyMem.gcAllocObject %0
    %2 = pyMem.initStr %1 to %arg0, %arg1
    return %2 : !py.dynamic
}

// CHECK-LABEL: @foo
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-NEXT: %[[STR:.*]] = llvm.mlir.addressof @builtins.str
// CHECK-NEXT: %[[STR_CAST:.*]] = llvm.bitcast %[[STR]]
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
// CHECK-NEXT: llvm.store %[[STR_CAST]], %[[GEP]]
// CHECK-NEXT: %[[STR_OBJ:.*]] = llvm.bitcast %[[RESULT]]
// CHECK-NEXT: %[[ZERO_I:.*]] = llvm.mlir.constant(0 : index)
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i32)
// CHECK-NEXT: %[[ARG0_STR:.*]] = llvm.bitcast %[[ARG0]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[ARG0_STR]][%[[ZERO]], %[[ONE]], %[[ZERO]]]
// CHECK-NEXT: %[[SIZE_0:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[SIZE_SUM_0:.*]] = llvm.add %[[ZERO_I]], %[[SIZE_0]]
// CHECK-NEXT: %[[ARG1_STR:.*]] = llvm.bitcast %[[ARG1]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[ARG1_STR]][%[[ZERO]], %[[ONE]], %[[ZERO]]]
// CHECK-NEXT: %[[SIZE_1:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.add %[[SIZE_SUM_0]], %[[SIZE_1]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[STR_OBJ]][%[[ZERO]], %[[ONE]], %[[ZERO]]]
// CHECK-NEXT: llvm.store %[[SIZE]], %[[GEP]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[STR_OBJ]][%[[ZERO]], %[[ONE]], %[[ONE]]]
// CHECK-NEXT: llvm.store %[[SIZE]], %[[GEP]]
// CHECK-NEXT: %[[ARRAY:.*]] = llvm.call @malloc(%[[SIZE]])
// CHECK-NEXT: %[[TWO:.*]] = llvm.mlir.constant(2 : i32)
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[STR_OBJ]][%[[ZERO]], %[[ONE]], %[[TWO]]]
// CHECK-NEXT: llvm.store %[[ARRAY]], %[[GEP]]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.mlir.constant(0 : index)

// CHECK-NEXT: %[[ARG0_STR:.*]] = llvm.bitcast %[[ARG0]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[ARG0_STR]][%[[ZERO]], %[[ONE]], %[[ZERO]]]
// CHECK-NEXT: %[[SIZE_0:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[ARG0_STR]][%[[ZERO]], %[[ONE]], %[[TWO]]]
// CHECK-NEXT: %[[ARRAY_0:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[DEST:.*]] = llvm.getelementptr %[[ARRAY]][%[[SIZE]]]
// CHECK-NEXT: %[[FALSE:.*]] = llvm.mlir.constant(false)
// CHECK-NEXT: "llvm.intr.memcpy"(%[[DEST]], %[[ARRAY_0]], %[[SIZE_0]], %[[FALSE]])
// CHECK-NEXT: %[[SIZE_NEW:.*]] = llvm.add %[[SIZE]], %[[SIZE_0]]

// CHECK-NEXT: %[[ARG1_STR:.*]] = llvm.bitcast %[[ARG1]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[ARG1_STR]][%[[ZERO]], %[[ONE]], %[[ZERO]]]
// CHECK-NEXT: %[[SIZE_1:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[ARG1_STR]][%[[ZERO]], %[[ONE]], %[[TWO]]]
// CHECK-NEXT: %[[ARRAY_1:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[DEST:.*]] = llvm.getelementptr %[[ARRAY]][%[[SIZE_NEW]]]
// CHECK-NEXT: %[[FALSE:.*]] = llvm.mlir.constant(false)
// CHECK-NEXT: "llvm.intr.memcpy"(%[[DEST]], %[[ARRAY_1]], %[[SIZE_1]], %[[FALSE]])
// CHECK-NEXT: llvm.add %[[SIZE_NEW]], %[[SIZE_1]]

// CHECK-NEXT: llvm.return %[[RESULT]]
