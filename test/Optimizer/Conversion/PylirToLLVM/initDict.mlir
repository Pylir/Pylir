// RUN: pylir-opt %s -convert-arith-to-llvm -convert-pylir-to-llvm --reconcile-unrealized-casts --split-input-file | FileCheck %s


py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.dict = #py.type
py.globalValue const @builtins.tuple = #py.type

func.func @foo() -> !py.dynamic {
    %0 = py.constant(#py.ref<@builtins.dict>)
    %c0 = arith.constant 0 : index
    %1 = pyMem.gcAllocObject %0[%c0]
    %2 = pyMem.initDict %1
    return %2 : !py.dynamic
}

// CHECK-LABEL: llvm.func @foo
// CHECK-NEXT: %[[DICT:.*]] = llvm.mlir.addressof @builtins.dict
// CHECK: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc
// CHECK: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][0, 0]
// CHECK-NEXT: llvm.store %[[DICT]], %[[GEP]]
// CHECK-NEXT: llvm.return %[[MEMORY]]
