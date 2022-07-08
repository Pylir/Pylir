// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s


py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.str = #py.type
py.globalValue const @builtins.tuple = #py.type

func.func @foo(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = py.constant(@builtins.str)
    %1 = pyMem.gcAllocObject %0
    %2 = pyMem.initStr %1 to %arg0, %arg1
    return %2 : !py.dynamic
}

// CHECK-LABEL: @foo
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-NEXT: %[[STR:.*]] = llvm.mlir.addressof @builtins.str
// CHECK-NEXT: %[[BYTES:.*]] = llvm.mlir.constant(32 : index)
// CHECK-NEXT: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc(%[[BYTES]])
// CHECK-NEXT: %[[ZERO_I8:.*]] = llvm.mlir.constant(0 : i8)
// CHECK-NEXT: %[[FALSE:.*]] = llvm.mlir.constant(false)
// CHECK-NEXT: "llvm.intr.memset"(%[[MEMORY]], %[[ZERO_I8]], %[[BYTES]], %[[FALSE]])
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][%[[ZERO]], 0]
// CHECK-NEXT: llvm.store %[[STR]], %[[GEP]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[BUFFER:.*]] = llvm.getelementptr %[[MEMORY]][%[[ZERO]], 1]
// CHECK-NEXT: %[[ZERO_I:.*]] = llvm.mlir.constant(0 : index)
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[ARG0]][%[[ZERO]], 1]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[GEP1]][%[[ZERO]], 0]
// CHECK-NEXT: %[[SIZE_0:.*]] = llvm.load %[[GEP2]]
// CHECK-NEXT: %[[SIZE_SUM_0:.*]] = llvm.add %[[ZERO_I]], %[[SIZE_0]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[ARG1]][%[[ZERO]], 1]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[GEP1]][%[[ZERO]], 0]
// CHECK-NEXT: %[[SIZE_1:.*]] = llvm.load %[[GEP2]]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.add %[[SIZE_SUM_0]], %[[SIZE_1]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[BUFFER]][%[[ZERO]], 0]
// CHECK-NEXT: llvm.store %[[SIZE]], %[[GEP]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[BUFFER]][%[[ZERO]], 1]
// CHECK-NEXT: llvm.store %[[SIZE]], %[[GEP]]
// CHECK-NEXT: %[[ARRAY:.*]] = llvm.call @malloc(%[[SIZE]])
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[BUFFER]][%[[ZERO]], 2]
// CHECK-NEXT: llvm.store %[[ARRAY]], %[[GEP]]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.mlir.constant(0 : index)
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[ARG0]][%[[ZERO]], 1]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[GEP1]][%[[ZERO]], 0]
// CHECK-NEXT: %[[SIZE_0:.*]] = llvm.load %[[GEP2]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[GEP1]][%[[ZERO]], 2]
// CHECK-NEXT: %[[ARRAY_0:.*]] = llvm.load %[[GEP2]]
// CHECK-NEXT: %[[DEST:.*]] = llvm.getelementptr %[[ARRAY]][%[[SIZE]]]
// CHECK-NEXT: %[[FALSE:.*]] = llvm.mlir.constant(false)
// CHECK-NEXT: "llvm.intr.memcpy"(%[[DEST]], %[[ARRAY_0]], %[[SIZE_0]], %[[FALSE]])
// CHECK-NEXT: %[[SIZE_NEW:.*]] = llvm.add %[[SIZE]], %[[SIZE_0]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[ARG1]][%[[ZERO]], 1]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[GEP1]][%[[ZERO]], 0]
// CHECK-NEXT: %[[SIZE_1:.*]] = llvm.load %[[GEP2]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[GEP1]][%[[ZERO]], 2]
// CHECK-NEXT: %[[ARRAY_1:.*]] = llvm.load %[[GEP2]]
// CHECK-NEXT: %[[DEST:.*]] = llvm.getelementptr %[[ARRAY]][%[[SIZE_NEW]]]
// CHECK-NEXT: %[[FALSE:.*]] = llvm.mlir.constant(false)
// CHECK-NEXT: "llvm.intr.memcpy"(%[[DEST]], %[[ARRAY_1]], %[[SIZE_1]], %[[FALSE]])
// CHECK-NEXT: llvm.add %[[SIZE_NEW]], %[[SIZE_1]]

// CHECK-NEXT: llvm.return %[[MEMORY]]
