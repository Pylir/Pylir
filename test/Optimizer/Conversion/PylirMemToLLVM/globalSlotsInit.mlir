// RUN: pylir-opt %s -convert-pylirMem-to-llvm --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type<slots: #py.slots<{"__slots__" to #py.tuple<(#py.str<"__slots__">,#py.str<"__eq__">,#py.str<"__hash__">)>}>>

py.globalValue const @builtins.object = #py.type // stub
py.globalValue const @builtins.str = #py.type // stub
py.globalValue const @builtins.tuple = #py.type // stub

// CHECK-LABEL: @builtins.type
// CHECK-NEXT: %[[UNDEF:.*]] = llvm.mlir.undef
// CHECK-NEXT: %[[TYPE:.*]] = llvm.mlir.addressof @builtins.type
// CHECK-NEXT: %[[BITCAST:.*]] = llvm.bitcast %1
// CHECK-NEXT: %[[UNDEF1:.*]] = llvm.insertvalue %[[BITCAST]], %[[UNDEF]][0 : i32]
// CHECK-NEXT: %[[NULL:.*]] = llvm.mlir.null
// CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i32)
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[NULL]][%[[ONE]]]
// CHECK-NEXT: %[[PTR_TO_INT:.*]] = llvm.ptrtoint %[[GEP]]
// CHECK-NEXT: %[[THREE:.*]] = llvm.mlir.constant(8 : i32)
// CHECK-NEXT: %[[UDIV:.*]] = llvm.udiv %[[PTR_TO_INT]], %[[THREE]]
// CHECK-NEXT: %[[UNDEF2:.*]] = llvm.insertvalue %[[UDIV]], %[[UNDEF1]][1 : i32]
// CHECK-NEXT: %[[ADDRESS:.*]] = llvm.mlir.addressof
// CHECK-NEXT: %[[BITCAST:.*]] = llvm.bitcast %[[ADDRESS]]
// CHECK-NEXT: %[[UNDEF3:.*]] = llvm.insertvalue %[[BITCAST]], %[[UNDEF2]][2 : i32, 0 : i32]
// CHECK-NEXT: %[[NULL:.*]] = llvm.mlir.null
// CHECK-NEXT: %[[UNDEF4:.*]] = llvm.insertvalue %[[NULL]], %[[UNDEF3]][2 : i32, 1 : i32]
// CHECK-NEXT: %[[NULL:.*]] = llvm.mlir.null
// CHECK-NEXT: %[[UNDEF5:.*]] = llvm.insertvalue %[[NULL]], %[[UNDEF4]][2 : i32, 2 : i32]
// CHECK-NEXT: llvm.return %[[UNDEF5]]
