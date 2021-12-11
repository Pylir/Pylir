// RUN: pylir-opt %s -convert-pylirMem-to-llvm --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type // stub
py.globalValue @builtins.function = #py.type // stub
py.globalValue @builtins.None = #py.type // stub

py.globalValue @foo = #py.function<@bar>

func @bar(%arg0 : !py.dynamic, %arg1 : !py.dynamic, %arg2 : !py.dynamic) -> !py.dynamic {
    %0 = py.constant @foo
    %1 = py.function.getFunction %0
    %2 = call_indirect %1(%0, %arg1, %arg2) : (!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic
    return %2 : !py.dynamic
}

// CHECK-LABEL: llvm.func @bar
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG2:[[:alnum:]]+]]
// CHECK-NEXT: %[[ADDRESS:.*]] = llvm.mlir.addressof @foo
// CHECK-NEXT: %[[CAST:.*]] = llvm.bitcast %[[ADDRESS]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i32)
// CHECK-NEXT: %[[FUNCTION:.*]] = llvm.bitcast %[[CAST]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[FUNCTION]][%[[ZERO]], %[[ONE]]]
// CHECK-NEXT: %[[LOADED:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[CALL:.*]] = llvm.call %[[LOADED]](%[[CAST]], %[[ARG1]], %[[ARG2]])
// CHECK-NEXT: llvm.return %[[CALL]]
