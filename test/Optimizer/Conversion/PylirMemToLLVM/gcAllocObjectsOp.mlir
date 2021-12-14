// RUN: pylir-opt %s -convert-pylirMem-to-llvm --split-input-file | FileCheck %s


py.globalValue const @builtins.type = #py.type<slots: #py.slots<{"__slots__" to #py.tuple<(#py.str<"__slots__">,#py.str<"__eq__">,#py.str<"__hash__">)>}>>

py.globalValue const @builtins.tuple = #py.type // stub
py.globalValue const @builtins.str = #py.type // stub

func @foo() -> !pyMem.memory {
    %0 = py.constant @builtins.tuple
    %1 = pyMem.gcAllocObject %0
    return %1 : !pyMem.memory
}

// CHECK-LABEL: llvm.func @foo
// CHECK-NEXT: %[[TUPLE:.*]] = llvm.mlir.addressof @builtins.tuple
// CHECK-NEXT: %[[TUPLE_CAST:.*]] = llvm.bitcast %[[TUPLE]]
// CHECK-NEXT: %[[NULL:.*]] = llvm.mlir.null
// CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i32)
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[NULL]][%[[ONE]]]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.ptrtoint %[[GEP]]
// CHECK-NEXT: %[[SLOTS:.*]] = llvm.mlir.constant(0 : index)
// CHECK-NEXT: %[[BYTES:.*]] = llvm.add %[[SIZE]], %[[SLOTS]]
// CHECK-NEXT: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc(%[[BYTES]])
// CHECK-NEXT: %[[RESULT:.*]] = llvm.bitcast %[[MEMORY]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[RESULT]][%[[ZERO]], %[[ZERO]]]
// CHECK-NEXT: llvm.store %[[TUPLE_CAST]], %[[GEP]]
// CHECK-NEXT: llvm.return %9

// -----

py.globalValue const @builtins.type = #py.type<slots: #py.slots<{"__slots__" to #py.tuple<(#py.str<"__slots__">,#py.str<"__eq__">,#py.str<"__hash__">)>}>>

py.globalValue const @builtins.tuple = #py.type // stub
py.globalValue const @builtins.str = #py.type // stub

func @foo(%arg0 : !py.dynamic) -> !pyMem.memory {
    %0 = pyMem.gcAllocObject %arg0
    return %0 : !pyMem.memory
}

// CHECK-LABEL: llvm.func @foo
