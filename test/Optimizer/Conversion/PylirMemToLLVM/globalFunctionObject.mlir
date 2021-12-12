// RUN: pylir-opt %s -convert-pylirMem-to-llvm --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type // stub
py.globalValue @builtins.function = #py.type // stub
py.globalValue @builtins.None = #py.type // stub

py.globalValue @foo = #py.function<@bar>

func @bar(%arg0 : !py.dynamic, %arg1 : !py.dynamic, %arg2 : !py.dynamic) -> !py.dynamic {
    return %arg0 : !py.dynamic
}

// CHECK-LABEL: @foo
// CHECK-NEXT: %[[UNDEF:.*]] = llvm.mlir.undef
// CHECK-NEXT: %[[TYPE:.*]] = llvm.mlir.addressof @builtins.function
// CHECK-NEXT: %[[BITCAST:.*]] = llvm.bitcast %1
// CHECK-NEXT: %[[UNDEF1:.*]] = llvm.insertvalue %[[BITCAST]], %[[UNDEF]][0 : i32]
// CHECK-NEXT: %[[ADDRESS:.*]] = llvm.mlir.addressof @bar
// CHECK-NEXT: %[[UNDEF2:.*]] = llvm.insertvalue %[[ADDRESS]], %[[UNDEF1]][1 : i32]
// CHECK-NEXT: llvm.return %[[UNDEF2]]