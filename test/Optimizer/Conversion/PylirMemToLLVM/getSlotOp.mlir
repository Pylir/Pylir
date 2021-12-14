// RUN: pylir-opt %s -convert-pylirMem-to-llvm --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type<slots: #py.slots<{"__slots__" to #py.tuple<(#py.str<"__slots__">)>}>>
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.tuple = #py.type

func @foo() -> !py.dynamic {
    %0 = py.constant @builtins.type
    %1 = py.constant @builtins.tuple
    %2 = py.getSlot "__slots__" from %1 : %0
    return %2 : !py.dynamic
}

// CHECK-LABEL: @foo
// CHECK-NEXT: %[[TYPE:.*]] = llvm.mlir.addressof @builtins.type
// CHECK-NEXT: %[[TYPE_CAST:.*]] = llvm.bitcast %[[TYPE]]
// CHECK-NEXT: %[[TUPLE:.*]] = llvm.mlir.addressof @builtins.tuple
// CHECK-NEXT: %[[TUPLE_CAST:.*]] = llvm.bitcast %[[TUPLE]]
// CHECK-NEXT: %[[NULL:.*]] = llvm.mlir.null
// CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i32)
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[NULL]][%[[ONE]]]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.ptrtoint %[[GEP]]
// CHECK-NEXT: %[[I8:.*]] = llvm.bitcast %[[TUPLE_CAST]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[I8]][%[[SIZE]]]
// CHECK-NEXT: %[[OBJECT_PTR:.*]] = llvm.bitcast %[[GEP]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : index)
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[OBJECT_PTR]][%[[ZERO]]]
// CHECK-NEXT: %[[LOAD:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: llvm.return %[[LOAD]]

// -----

py.globalValue @builtins.type = #py.type<slots: #py.slots<{"__slots__" to #py.tuple<(#py.str<"__slots__">)>}>>
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type

func @foo(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = py.getSlot "__slots__" from %arg0 : %arg1
    return %0 : !py.dynamic
}

// CHECK-LABEL: llvm.func @foo
