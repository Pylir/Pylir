// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s


py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.str = #py.type
py.globalValue const @builtins.tuple = #py.type

func @foo(%arg0 : !py.unknown, %arg1 : !py.unknown) -> !py.unknown {
    %0 = py.constant(@builtins.str) : !py.unknown
    %1 = pyMem.gcAllocObject %0 : !py.unknown
    %2 = pyMem.initStr %1 to %arg0, %arg1 : (!py.unknown, !py.unknown) -> !py.unknown
    return %2 : !py.unknown
}

// CHECK-LABEL: @foo
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-NEXT: %[[STR:.*]] = llvm.mlir.addressof @builtins.str
// CHECK-NEXT: %[[STR_CAST:.*]] = llvm.bitcast %[[STR]]
// CHECK-NEXT: %[[BYTES:.*]] = llvm.mlir.constant(32 : index)
// CHECK-NEXT: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc(%[[BYTES]])
// CHECK-NEXT: %[[ZERO_I8:.*]] = llvm.mlir.constant(0 : i8)
// CHECK-NEXT: %[[FALSE:.*]] = llvm.mlir.constant(false)
// CHECK-NEXT: "llvm.intr.memset"(%[[MEMORY]], %[[ZERO_I8]], %[[BYTES]], %[[FALSE]])
// CHECK-NEXT: %[[RESULT:.*]] = llvm.bitcast %[[MEMORY]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[RESULT]][0, 0]
// CHECK-NEXT: llvm.store %[[STR_CAST]], %[[GEP]]
// CHECK-NEXT: %[[STR_OBJ:.*]] = llvm.bitcast %[[RESULT]]
// CHECK-NEXT: %[[BUFFER:.*]] = llvm.getelementptr %[[STR_OBJ]][0, 1]
// CHECK-NEXT: %[[ZERO_I:.*]] = llvm.mlir.constant(0 : index)
// CHECK-NEXT: %[[ARG0_STR:.*]] = llvm.bitcast %[[ARG0]]
// CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[ARG0_STR]][0, 1]
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[GEP1]][0, 0]
// CHECK-NEXT: %[[SIZE_0:.*]] = llvm.load %[[GEP2]]
// CHECK-NEXT: %[[SIZE_SUM_0:.*]] = llvm.add %[[ZERO_I]], %[[SIZE_0]]
// CHECK-NEXT: %[[ARG1_STR:.*]] = llvm.bitcast %[[ARG1]]
// CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[ARG1_STR]][0, 1]
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[GEP1]][0, 0]
// CHECK-NEXT: %[[SIZE_1:.*]] = llvm.load %[[GEP2]]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.add %[[SIZE_SUM_0]], %[[SIZE_1]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[BUFFER]][0, 0]
// CHECK-NEXT: llvm.store %[[SIZE]], %[[GEP]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[BUFFER]][0, 1]
// CHECK-NEXT: llvm.store %[[SIZE]], %[[GEP]]
// CHECK-NEXT: %[[ARRAY:.*]] = llvm.call @malloc(%[[SIZE]])
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[BUFFER]][0, 2]
// CHECK-NEXT: llvm.store %[[ARRAY]], %[[GEP]]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.mlir.constant(0 : index)

// CHECK-NEXT: %[[ARG0_STR:.*]] = llvm.bitcast %[[ARG0]]
// CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[ARG0_STR]][0, 1]
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[GEP1]][0, 0]
// CHECK-NEXT: %[[SIZE_0:.*]] = llvm.load %[[GEP2]]
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[GEP1]][0, 2]
// CHECK-NEXT: %[[ARRAY_0:.*]] = llvm.load %[[GEP2]]
// CHECK-NEXT: %[[DEST:.*]] = llvm.getelementptr %[[ARRAY]][%[[SIZE]]]
// CHECK-NEXT: %[[FALSE:.*]] = llvm.mlir.constant(false)
// CHECK-NEXT: "llvm.intr.memcpy"(%[[DEST]], %[[ARRAY_0]], %[[SIZE_0]], %[[FALSE]])
// CHECK-NEXT: %[[SIZE_NEW:.*]] = llvm.add %[[SIZE]], %[[SIZE_0]]

// CHECK-NEXT: %[[ARG1_STR:.*]] = llvm.bitcast %[[ARG1]]
// CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[ARG1_STR]][0, 1]
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[GEP1]][0, 0]
// CHECK-NEXT: %[[SIZE_1:.*]] = llvm.load %[[GEP2]]
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[GEP1]][0, 2]
// CHECK-NEXT: %[[ARRAY_1:.*]] = llvm.load %[[GEP2]]
// CHECK-NEXT: %[[DEST:.*]] = llvm.getelementptr %[[ARRAY]][%[[SIZE_NEW]]]
// CHECK-NEXT: %[[FALSE:.*]] = llvm.mlir.constant(false)
// CHECK-NEXT: "llvm.intr.memcpy"(%[[DEST]], %[[ARRAY_1]], %[[SIZE_1]], %[[FALSE]])
// CHECK-NEXT: llvm.add %[[SIZE_NEW]], %[[SIZE_1]]

// CHECK-NEXT: llvm.return %[[RESULT]]
