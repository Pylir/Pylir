// RUN: pylir-opt %s -convert-pylirMem-to-llvm --split-input-file | FileCheck %s

py.globalValue @builtins.tuple = #py.tuple<()>

// CHECK: llvm.mlir.global private unnamed_addr constant @[[BUFFER:buffer\$[[:alnum:]]*]]()
// CHECK-NEXT: %[[UNDEF:.*]] = llvm.mlir.undef
// CHECK-NEXT: llvm.return %[[UNDEF]]

// CHECK-LABEL: llvm.mlir.global external @builtins.tuple
// CHECK-NEXT: %[[UNDEF:.*]] = llvm.mlir.undef
// CHECK-NEXT: %[[TYPE:.*]] = llvm.mlir.addressof @builtins.tuple
// CHECK-NEXT: %[[CAST:.*]] = llvm.bitcast %[[TYPE]]
// CHECK-NEXT: %[[UNDEF1:.*]] = llvm.insertvalue %[[CAST]], %[[UNDEF]][0 : i32]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.mlir.constant
// CHECK-SAME: 0
// CHECK-NEXT: %[[UNDEF2:.*]] = llvm.insertvalue %[[SIZE]], %[[UNDEF1]][1 : i32, 0 : i32]
// CHECK-NEXT: %[[UNDEF3:.*]] = llvm.insertvalue %[[SIZE]], %[[UNDEF2]][1 : i32, 1 : i32]
// CHECK-NEXT: %[[BUFFER_ADDR:.*]] = llvm.mlir.addressof @[[BUFFER]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[BUFFER_ADDR]][%[[ZERO]], %[[ZERO]]]
// CHECK-NEXT: %[[UNDEF4:.*]] = llvm.insertvalue %[[GEP]], %[[UNDEF3]][1 : i32, 2 : i32]
// CHECK-NEXT: llvm.return %[[UNDEF4]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type

py.globalValue @foo = #py.str<"test">

// CHECK: llvm.mlir.global private unnamed_addr constant @[[BUFFER:buffer\$[[:alnum:]]*]]("test")

// CHECK-LABEL: llvm.mlir.global external @foo
// CHECK-NEXT: %[[UNDEF:.*]] = llvm.mlir.undef
// CHECK-NEXT: %[[TYPE:.*]] = llvm.mlir.addressof @builtins.str
// CHECK-NEXT: %[[CAST:.*]] = llvm.bitcast %[[TYPE]]
// CHECK-NEXT: %[[UNDEF1:.*]] = llvm.insertvalue %[[CAST]], %[[UNDEF]][0 : i32]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.mlir.constant(4 : i64)
// CHECK-NEXT: %[[UNDEF2:.*]] = llvm.insertvalue %[[SIZE]], %[[UNDEF1]][1 : i32, 0 : i32]
// CHECK-NEXT: %[[UNDEF3:.*]] = llvm.insertvalue %[[SIZE]], %[[UNDEF2]][1 : i32, 1 : i32]
// CHECK-NEXT: %[[BUFFER_ADDR:.*]] = llvm.mlir.addressof @[[BUFFER]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[BUFFER_ADDR]][%[[ZERO]], %[[ZERO]]]
// CHECK-NEXT: %[[UNDEF4:.*]] = llvm.insertvalue %[[GEP]], %[[UNDEF3]][1 : i32, 2 : i32]
// CHECK-NEXT: llvm.return %[[UNDEF4]]
