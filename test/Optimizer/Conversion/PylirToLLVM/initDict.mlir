// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s


py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.dict = #py.type
py.globalValue const @builtins.tuple = #py.type

func.func @foo() -> !py.dynamic {
    %0 = py.constant(@builtins.dict)
    %1 = pyMem.gcAllocObject %0
    %2 = pyMem.initDict %1
    return %2 : !py.dynamic
}

// CHECK-LABEL: llvm.func @foo
// CHECK-NEXT: %[[DICT:.*]] = llvm.mlir.addressof @builtins.dict
// CHECK-NEXT: %[[BYTES:.*]] = llvm.mlir.constant(48 : index)
// CHECK-NEXT: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc(%[[BYTES]])
// CHECK-NEXT: %[[ZERO_I8:.*]] = llvm.mlir.constant(0 : i8)
// CHECK-NEXT: %[[FALSE:.*]] = llvm.mlir.constant(false)
// CHECK-NEXT: "llvm.intr.memset"(%[[MEMORY]], %[[ZERO_I8]], %[[BYTES]], %[[FALSE]])
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][0, 0]
// CHECK-NEXT: llvm.store %[[DICT]], %[[GEP]]
// CHECK-NEXT: llvm.return %[[MEMORY]]
