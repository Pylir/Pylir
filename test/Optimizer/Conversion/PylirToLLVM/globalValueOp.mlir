// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.object = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @foo = #py.tuple<()>
py.globalValue @bar = #py.tuple<(@foo, @builtins.type)>

// CHECK-LABEL: llvm.mlir.global external constant @foo
// CHECK-NEXT: %[[UNDEF:.*]] = llvm.mlir.undef
// CHECK-NEXT: %[[TYPE:.*]] = llvm.mlir.addressof @builtins.tuple
// CHECK-NEXT: %[[CAST:.*]] = llvm.bitcast %[[TYPE]]
// CHECK-NEXT: %[[UNDEF1:.*]] = llvm.insertvalue %[[CAST]], %[[UNDEF]][0 : i32]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.mlir.constant(0 : i{{.*}})
// CHECK-NEXT: %[[UNDEF2:.*]] = llvm.insertvalue %[[SIZE]], %[[UNDEF1]][1 : i32]
// CHECK-NEXT: llvm.return %[[UNDEF2]]

// CHECK-LABEL: llvm.mlir.global external constant @bar
// CHECK-NEXT: %[[UNDEF:.*]] = llvm.mlir.undef
// CHECK-NEXT: %[[TYPE:.*]] = llvm.mlir.addressof @builtins.tuple
// CHECK-NEXT: %[[CAST:.*]] = llvm.bitcast %[[TYPE]]
// CHECK-NEXT: %[[UNDEF1:.*]] = llvm.insertvalue %[[CAST]], %[[UNDEF]][0 : i32]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.mlir.constant(2 : i{{.*}})
// CHECK-NEXT: %[[UNDEF2:.*]] = llvm.insertvalue %[[SIZE]], %[[UNDEF1]][1 : i32]
// CHECK-NEXT: %[[ADDRESS:.*]] = llvm.mlir.addressof @foo
// CHECK-NEXT: %[[CAST:.*]] = llvm.bitcast %[[ADDRESS]]
// CHECK-NEXT: %[[UNDEF3:.*]] = llvm.insertvalue %[[CAST]], %[[UNDEF2]][2 : i32, 0 : i32]
// CHECK-NEXT: %[[ADDRESS:.*]] = llvm.mlir.addressof @builtins.type
// CHECK-NEXT: %[[CAST:.*]] = llvm.bitcast %[[ADDRESS]]
// CHECK-NEXT: %[[UNDEF4:.*]] = llvm.insertvalue %[[CAST]], %[[UNDEF3]][2 : i32, 1 : i32]
// CHECK-NEXT: llvm.return %[[UNDEF4]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type

py.globalValue @foo = #py.str<"test">

// CHECK: llvm.mlir.global private unnamed_addr constant @[[BUFFER:buffer\$[[:alnum:]]*]]("test")

// CHECK-LABEL: llvm.mlir.global external constant @foo
// CHECK-NEXT: %[[UNDEF:.*]] = llvm.mlir.undef
// CHECK-NEXT: %[[TYPE:.*]] = llvm.mlir.addressof @builtins.str
// CHECK-NEXT: %[[CAST:.*]] = llvm.bitcast %[[TYPE]]
// CHECK-NEXT: %[[UNDEF1:.*]] = llvm.insertvalue %[[CAST]], %[[UNDEF]][0 : i32]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.mlir.constant(4 : i64)
// CHECK-NEXT: %[[UNDEF2:.*]] = llvm.insertvalue %[[SIZE]], %[[UNDEF1]][1 : i32, 0 : i32]
// CHECK-NEXT: %[[UNDEF3:.*]] = llvm.insertvalue %[[SIZE]], %[[UNDEF2]][1 : i32, 1 : i32]
// CHECK-NEXT: %[[BUFFER_ADDR:.*]] = llvm.mlir.addressof @[[BUFFER]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[BUFFER_ADDR]][0, 0]
// CHECK-NEXT: %[[UNDEF4:.*]] = llvm.insertvalue %[[GEP]], %[[UNDEF3]][1 : i32, 2 : i32]
// CHECK-NEXT: llvm.return %[[UNDEF4]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.list = #py.type

py.globalValue @bar = #py.list<[]>
py.globalValue const @foo = #py.list<[]>
py.globalValue "private" @foobar = #py.list<[]>
py.globalValue "private" const @barfoo = #py.list<[]>

// CHECK: llvm.mlir.global external @bar
// CHECK: llvm.mlir.global external constant @foo
// CHECK: llvm.mlir.global internal @foobar
// CHECK: llvm.mlir.global internal constant @barfoo

// -----

py.globalValue "private" const @sys.__excepthook__

// CHECK: llvm.mlir.global external constant @sys.__excepthook__()
