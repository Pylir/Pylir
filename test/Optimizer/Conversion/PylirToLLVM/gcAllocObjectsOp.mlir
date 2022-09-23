// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s


py.globalValue const @builtins.type = #py.type<slots = {__slots__ = #py.tuple<(#py.str<"__slots__">,#py.str<"__eq__">,#py.str<"__hash__">)>}>

py.globalValue const @builtins.tuple = #py.type // stub
py.globalValue const @builtins.str = #py.type // stub

func.func @foo() -> !pyMem.memory {
    %0 = py.constant(#py.ref<@builtins.str>)
    %1 = pyMem.gcAllocObject %0
    return %1 : !pyMem.memory
}

// CHECK-LABEL: llvm.func @foo
// CHECK-NEXT: %[[STR:.*]] = llvm.mlir.addressof @builtins.str
// CHECK-NEXT: %[[BYTES:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc(%[[BYTES]])
// CHECK-NEXT: %[[ZERO_I8:.*]] = llvm.mlir.constant(0 : i8)
// CHECK-NEXT: %[[FALSE:.*]] = llvm.mlir.constant(false)
// CHECK-NEXT: "llvm.intr.memset"(%[[MEMORY]], %[[ZERO_I8]], %[[BYTES]], %[[FALSE]])
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][0, 0]
// CHECK-NEXT: llvm.store %[[STR]], %[[GEP]]
// CHECK-NEXT: llvm.return %[[MEMORY]]

// -----

py.globalValue const @builtins.type = #py.type<slots = {__slots__ = #py.tuple<(#py.str<"__slots__">,#py.str<"__eq__">,#py.str<"__hash__">)>}>

py.globalValue const @builtins.tuple = #py.type // stub
py.globalValue const @builtins.str = #py.type // stub

func.func @foo(%arg0 : !py.dynamic) -> !pyMem.memory {
    %0 = pyMem.gcAllocObject %arg0
    return %0 : !pyMem.memory
}

// CHECK-LABEL: llvm.func @foo
