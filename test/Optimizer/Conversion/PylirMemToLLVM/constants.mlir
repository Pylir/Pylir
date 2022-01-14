// RUN: pylir-opt %s -convert-pylirMem-to-llvm --split-input-file | FileCheck %s

py.globalValue @builtins.tuple = #py.tuple<()>

func @constants() -> !py.dynamic {
    %0 = py.constant #py.tuple<()>
    return %0 : !py.dynamic
}

// CHECK: llvm.mlir.global private unnamed_addr constant @[[CONSTANT:const\$[[:alnum:]]*]]
// CHECK-NEXT: %[[UNDEF:.*]] = llvm.mlir.undef
// CHECK-NEXT: %[[TYPE:.*]] = llvm.mlir.addressof @builtins.tuple
// CHECK-NEXT: %[[CAST:.*]] = llvm.bitcast %[[TYPE]]
// CHECK-NEXT: %[[UNDEF1:.*]] = llvm.insertvalue %[[CAST]], %[[UNDEF]][0 : i32]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.mlir.constant
// CHECK-SAME: 0
// CHECK-NEXT: %[[UNDEF2:.*]] = llvm.insertvalue %[[SIZE]], %[[UNDEF1]][1 : i32, 0 : i32]
// CHECK-NEXT: %[[UNDEF3:.*]] = llvm.insertvalue %[[SIZE]], %[[UNDEF2]][1 : i32, 1 : i32]
// CHECK-NEXT: %[[BUFFER_ADDR:.*]] = llvm.mlir.addressof @[[BUFFER:buffer\$[[:alnum:]]*]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[BUFFER_ADDR]][0, 0]
// CHECK-NEXT: %[[UNDEF4:.*]] = llvm.insertvalue %[[GEP]], %[[UNDEF3]][1 : i32, 2 : i32]
// CHECK-NEXT: llvm.return %[[UNDEF4]]

// CHECK: llvm.mlir.global private unnamed_addr constant @[[BUFFER]]()
// CHECK-NEXT: %[[UNDEF:.*]] = llvm.mlir.undef
// CHECK-NEXT: llvm.return %[[UNDEF]]

// CHECK-LABEL: @constants
// CHECK-NEXT: %[[CONSTANT_ADDRESS:.*]] = llvm.mlir.addressof @[[CONSTANT]]
// CHECK-NEXT: %[[CASTED:.*]] = llvm.bitcast %[[CONSTANT_ADDRESS]]
// CHECK-NEXT: llvm.return %[[CASTED]]
