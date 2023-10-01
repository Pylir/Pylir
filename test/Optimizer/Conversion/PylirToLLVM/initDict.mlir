// RUN: pylir-opt %s -convert-arith-to-llvm -convert-pylir-to-llvm --reconcile-unrealized-casts --split-input-file | FileCheck %s


#builtins_type = #py.globalValue<builtins.type, const, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_dict = #py.globalValue<builtins.dict, const, initializer = #py.type>
py.external @builtins.dict, #builtins_dict
#builtins_tuple = #py.globalValue<builtins.tuple, const, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple

py.func @foo() -> !py.dynamic {
    %0 = constant(#builtins_dict)
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
