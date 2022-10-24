// RUN: pylir-opt %s -convert-arith-to-llvm -convert-pylir-to-llvm --reconcile-unrealized-casts --split-input-file | FileCheck %s


py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.str = #py.type
py.globalValue const @builtins.tuple = #py.type

func.func @foo(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = py.constant(#py.ref<@builtins.str>)
    %c0 = arith.constant 0 : index
    %1 = pyMem.gcAllocObject %0[%c0]
    %2 = pyMem.initStr %1 to %arg0, %arg1
    return %2 : !py.dynamic
}

// CHECK-LABEL: @foo
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-NEXT: %[[STR:.*]] = llvm.mlir.addressof @builtins.str
// CHECK: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc
// CHECK: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][0, 0]
// CHECK-NEXT: llvm.store %[[STR]], %[[GEP]]
// CHECK-NEXT: %[[BUFFER:.*]] = llvm.getelementptr %[[MEMORY]][0, 1]
// CHECK-NEXT: %[[ZERO_I:.*]] = llvm.mlir.constant(0 : index)
// CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[ARG0]][0, 1]
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[GEP1]][0, 0]
// CHECK-NEXT: %[[SIZE_0:.*]] = llvm.load %[[GEP2]]
// CHECK-NEXT: %[[SIZE_SUM_0:.*]] = llvm.add %[[ZERO_I]], %[[SIZE_0]]
// CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[ARG1]][0, 1]
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

// CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[ARG0]][0, 1]
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[GEP1]][0, 0]
// CHECK-NEXT: %[[SIZE_0:.*]] = llvm.load %[[GEP2]]
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[GEP1]][0, 2]
// CHECK-NEXT: %[[ARRAY_0:.*]] = llvm.load %[[GEP2]]
// CHECK-NEXT: %[[FALSE:.*]] = llvm.mlir.constant(false)
// CHECK-NEXT: "llvm.intr.memcpy"(%[[ARRAY]], %[[ARRAY_0]], %[[SIZE_0]], %[[FALSE]])
// CHECK-NEXT: %[[SIZE_NEW:.*]] = llvm.add %[[SIZE]], %[[SIZE_0]]

// CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[ARG1]][0, 1]
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[GEP1]][0, 0]
// CHECK-NEXT: %[[SIZE_1:.*]] = llvm.load %[[GEP2]]
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[GEP1]][0, 2]
// CHECK-NEXT: %[[ARRAY_1:.*]] = llvm.load %[[GEP2]]
// CHECK-NEXT: %[[DEST:.*]] = llvm.getelementptr %[[ARRAY]][%[[SIZE_NEW]]]
// CHECK-NEXT: %[[FALSE:.*]] = llvm.mlir.constant(false)
// CHECK-NEXT: "llvm.intr.memcpy"(%[[DEST]], %[[ARRAY_1]], %[[SIZE_1]], %[[FALSE]])
// CHECK-NEXT: llvm.add %[[SIZE_NEW]], %[[SIZE_1]]

// CHECK-NEXT: llvm.return %[[MEMORY]]
