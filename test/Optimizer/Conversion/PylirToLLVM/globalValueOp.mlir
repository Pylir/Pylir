// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.object = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @foo = #py.tuple<()>
py.globalValue @bar = #py.tuple<(@foo, @builtins.type)>

// CHECK-LABEL: llvm.mlir.global external constant @foo
// CHECK-NEXT: %[[UNDEF:.*]] = llvm.mlir.undef
// CHECK-NEXT: %[[TYPE:.*]] = llvm.mlir.addressof @builtins.tuple
// CHECK-NEXT: %[[UNDEF1:.*]] = llvm.insertvalue %[[TYPE]], %[[UNDEF]][0]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.mlir.constant(0 : i{{.*}})
// CHECK-NEXT: %[[UNDEF2:.*]] = llvm.insertvalue %[[SIZE]], %[[UNDEF1]][1]
// CHECK-NEXT: llvm.return %[[UNDEF2]]

// CHECK-LABEL: llvm.mlir.global external constant @bar
// CHECK-NEXT: %[[UNDEF:.*]] = llvm.mlir.undef
// CHECK-NEXT: %[[TYPE:.*]] = llvm.mlir.addressof @builtins.tuple
// CHECK-NEXT: %[[UNDEF1:.*]] = llvm.insertvalue %[[TYPE]], %[[UNDEF]][0]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.mlir.constant(2 : i{{.*}})
// CHECK-NEXT: %[[UNDEF2:.*]] = llvm.insertvalue %[[SIZE]], %[[UNDEF1]][1]
// CHECK-NEXT: %[[ADDRESS:.*]] = llvm.mlir.addressof @foo
// CHECK-NEXT: %[[UNDEF3:.*]] = llvm.insertvalue %[[ADDRESS]], %[[UNDEF2]][2, 0]
// CHECK-NEXT: %[[ADDRESS:.*]] = llvm.mlir.addressof @builtins.type
// CHECK-NEXT: %[[UNDEF4:.*]] = llvm.insertvalue %[[ADDRESS]], %[[UNDEF3]][2, 1]
// CHECK-NEXT: llvm.return %[[UNDEF4]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.tuple = #py.type

py.globalValue @foo = #py.str<"test">

// CHECK: llvm.mlir.global private unnamed_addr constant @[[BUFFER:buffer\$[[:alnum:]]*]]("test")

// CHECK-LABEL: llvm.mlir.global external constant @foo
// CHECK-NEXT: %[[UNDEF:.*]] = llvm.mlir.undef
// CHECK-NEXT: %[[TYPE:.*]] = llvm.mlir.addressof @builtins.str
// CHECK-NEXT: %[[UNDEF1:.*]] = llvm.insertvalue %[[TYPE]], %[[UNDEF]][0]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.mlir.constant(4 : i64)
// CHECK-NEXT: %[[UNDEF2:.*]] = llvm.insertvalue %[[SIZE]], %[[UNDEF1]][1, 0]
// CHECK-NEXT: %[[UNDEF3:.*]] = llvm.insertvalue %[[SIZE]], %[[UNDEF2]][1, 1]
// CHECK-NEXT: %[[BUFFER_ADDR:.*]] = llvm.mlir.addressof @[[BUFFER]]
// CHECK-NEXT: %[[UNDEF4:.*]] = llvm.insertvalue %[[BUFFER_ADDR]], %[[UNDEF3]][1, 2]
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
