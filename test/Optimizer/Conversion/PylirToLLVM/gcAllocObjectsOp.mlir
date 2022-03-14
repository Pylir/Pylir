// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s


py.globalValue const @builtins.type = #py.type<slots = {__slots__ = #py.tuple<value = (#py.str<value = "__slots__">,#py.str<value = "__eq__">,#py.str<value = "__hash__">)>}>

py.globalValue const @builtins.tuple = #py.type<> // stub
py.globalValue const @builtins.str = #py.type<> // stub

func @foo() -> !pyMem.memory {
    %0 = py.constant @builtins.str
    %1 = pyMem.gcAllocObject %0
    return %1 : !pyMem.memory
}

// CHECK-LABEL: llvm.func @foo
// CHECK-NEXT: %[[STR:.*]] = llvm.mlir.addressof @builtins.str
// CHECK-NEXT: %[[STR_CAST:.*]] = llvm.bitcast %[[STR]]
// CHECK-NEXT: %[[BYTES:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc(%[[BYTES]])
// CHECK-NEXT: %[[ZERO_I8:.*]] = llvm.mlir.constant(0 : i8)
// CHECK-NEXT: %[[FALSE:.*]] = llvm.mlir.constant(false)
// CHECK-NEXT: "llvm.intr.memset"(%[[MEMORY]], %[[ZERO_I8]], %[[BYTES]], %[[FALSE]])
// CHECK-NEXT: %[[RESULT:.*]] = llvm.bitcast %[[MEMORY]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[RESULT]][0, 0]
// CHECK-NEXT: llvm.store %[[STR_CAST]], %[[GEP]]
// CHECK-NEXT: llvm.return %[[RESULT]]

// -----

py.globalValue const @builtins.type = #py.type<slots = {__slots__ = #py.tuple<value = (#py.str<value = "__slots__">,#py.str<value = "__eq__">,#py.str<value = "__hash__">)>}>

py.globalValue const @builtins.tuple = #py.type<> // stub
py.globalValue const @builtins.str = #py.type<> // stub

func @foo(%arg0 : !py.dynamic) -> !pyMem.memory {
    %0 = pyMem.gcAllocObject %arg0
    return %0 : !pyMem.memory
}

// CHECK-LABEL: llvm.func @foo
