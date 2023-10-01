// RUN: pylir-opt %s -convert-arith-to-llvm -convert-pylir-to-llvm --reconcile-unrealized-casts --split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, const, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_float = #py.globalValue<builtins.float, const, initializer = #py.type>
py.external @builtins.float, #builtins_float
#builtins_tuple = #py.globalValue<builtins.tuple, const, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple

py.func @foo(%value : f64) -> !py.dynamic {
    %0 = constant(#builtins_float)
    %c0 = arith.constant 0 : index
    %1 = pyMem.gcAllocObject %0[%c0]
    %2 = pyMem.initFloat %1 to %value
    return %2 : !py.dynamic
}

// CHECK-DAG: #[[DESC:.*]] = #llvm.tbaa_type_desc<id = "Python Float Value"{{.*}}>
// CHECK-DAG: #[[$PYTHON_FLOAT_VALUE:.*]] = #llvm.tbaa_tag<base_type = #[[DESC]], access_type = #[[DESC]], offset = 0>

// CHECK-LABEL: llvm.func @foo
// CHECK-SAME: %[[VALUE:[[:alnum:]]+]]
// CHECK: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc
// CHECK: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][0, 1]
// CHECK-NEXT: llvm.store %[[VALUE]], %[[GEP]] {tbaa = [#[[$PYTHON_FLOAT_VALUE]]]}
// CHECK-NEXT: llvm.return %[[MEMORY]]
