// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.globalValue @builtins.tuple = #py.tuple<value = ()>

func @constants() -> !py.unknown {
    %0 = py.constant(#py.tuple<value = ()>) : !py.unknown
    return %0 : !py.unknown
}

// CHECK: llvm.mlir.global private unnamed_addr constant @[[CONSTANT:const\$[[:alnum:]]*]]
// CHECK-NEXT: %[[UNDEF:.*]] = llvm.mlir.undef
// CHECK-NEXT: %[[TYPE:.*]] = llvm.mlir.addressof @builtins.tuple
// CHECK-NEXT: %[[CAST:.*]] = llvm.bitcast %[[TYPE]]
// CHECK-NEXT: %[[UNDEF1:.*]] = llvm.insertvalue %[[CAST]], %[[UNDEF]][0 : i32]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.mlir.constant(0 : i{{.*}})
// CHECK-NEXT: %[[UNDEF2:.*]] = llvm.insertvalue %[[SIZE]], %[[UNDEF1]][1 : i32]
// CHECK-NEXT: llvm.return %[[UNDEF2]]

// CHECK-LABEL: @constants
// CHECK-NEXT: %[[CONSTANT_ADDRESS:.*]] = llvm.mlir.addressof @[[CONSTANT]]
// CHECK-NEXT: %[[CASTED:.*]] = llvm.bitcast %[[CONSTANT_ADDRESS]]
// CHECK-NEXT: llvm.return %[[CASTED]]
