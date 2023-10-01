// RUN: pylir-opt %s -convert-arith-to-llvm -convert-pylir-to-llvm --reconcile-unrealized-casts --split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, const, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, const, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_list = #py.globalValue<builtins.list, const, initializer = #py.type>
py.external @builtins.list, #builtins_list

py.func @foo() -> !py.dynamic {
    %0 = constant(#builtins_list)
    %c0 = arith.constant 0 : index
    %1 = pyMem.gcAllocObject %0[%c0]
    %2 = pyMem.initList %1 to [%0]
    return %2 : !py.dynamic
}

// CHECK-DAG: #[[DESC:.*]] = #llvm.tbaa_type_desc<id = "Python List Size"{{.*}}>
// CHECK-DAG: #[[$PYTHON_LIST_SIZE:.*]] = #llvm.tbaa_tag<base_type = #[[DESC]], access_type = #[[DESC]], offset = 0>
// CHECK-DAG: #[[DESC:.*]] = #llvm.tbaa_type_desc<id = "Python Tuple Size"{{.*}}>
// CHECK-DAG: #[[$PYTHON_TUPLE_SIZE:.*]] = #llvm.tbaa_tag<base_type = #[[DESC]], access_type = #[[DESC]], offset = 0>
// CHECK-DAG: #[[DESC:.*]] = #llvm.tbaa_type_desc<id = "Python Tuple Elements"{{.*}}>
// CHECK-DAG: #[[$PYTHON_TUPLE_ELEMENTS:.*]] = #llvm.tbaa_tag<base_type = #[[DESC]], access_type = #[[DESC]], offset = 0>
// CHECK-DAG: #[[DESC:.*]] = #llvm.tbaa_type_desc<id = "Python List Tuple"{{.*}}>
// CHECK-DAG: #[[$PYTHON_LIST_TUPLE:.*]] = #llvm.tbaa_tag<base_type = #[[DESC]], access_type = #[[DESC]], offset = 0>

// CHECK-LABEL: llvm.func @foo
// CHECK-NEXT: %[[LIST:.*]] = llvm.mlir.addressof @builtins.list
// CHECK: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc
// CHECK-NEXT: %[[ZERO_I8:.*]] = llvm.mlir.constant(0 : i8)
// CHECK: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][0, 0]
// CHECK-NEXT: llvm.store %[[LIST]], %[[GEP]]
// CHECK-NEXT: %[[LEN:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[SIZE_PTR:.*]] = llvm.getelementptr %[[MEMORY]][0, 1]
// CHECK-NEXT: llvm.store %[[LEN]], %[[SIZE_PTR]] {tbaa = [#[[$PYTHON_LIST_SIZE]]]}
// CHECK-NEXT: %[[TUPLE_TYPE:.*]] = llvm.mlir.addressof @builtins.tuple
// CHECK-NEXT: %[[HEADER_SIZE:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[ELEMENT_SIZE:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[TRAILING_SIZE:.*]] = llvm.mul %[[LEN]], %[[ELEMENT_SIZE]]
// CHECK-NEXT: %[[BYTES:.*]] = llvm.add %[[TRAILING_SIZE]], %[[HEADER_SIZE]]
// CHECK-NEXT: %[[TUPLE_MEMORY:.*]] = llvm.call @pylir_gc_alloc(%[[BYTES]])
// CHECK-NEXT: %[[ZERO_I8:.*]] = llvm.mlir.constant(0 : i8)
// CHECK-NEXT: "llvm.intr.memset"(%[[TUPLE_MEMORY]], %[[ZERO_I8]], %[[BYTES]])
// CHECK-SAME: isVolatile = false
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TUPLE_MEMORY]][0, 0]
// CHECK-NEXT: llvm.store %[[TUPLE_TYPE]], %[[GEP]]
// CHECK-NEXT: %[[CAPACITY:.*]] = llvm.mlir.constant(1 : i{{.*}})
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TUPLE_MEMORY]][0, 1]
// CHECK-NEXT: llvm.store %[[CAPACITY]], %[[GEP]] {tbaa = [#[[$PYTHON_TUPLE_SIZE]]]}
// CHECK-NEXT: %[[TRAILING:.*]] = llvm.getelementptr %[[TUPLE_MEMORY]][0, 2]
// CHECK-NEXT: %[[FIRST:.*]] = llvm.getelementptr %[[TRAILING]][0, 0]
// CHECK-NEXT: llvm.store %[[LIST]], %[[FIRST]] {tbaa = [#[[$PYTHON_TUPLE_ELEMENTS]]]}
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][0, 2]
// CHECK-NEXT: llvm.store %[[TUPLE_MEMORY]], %[[GEP]] {tbaa = [#[[$PYTHON_LIST_TUPLE]]]}
// CHECK-NEXT: llvm.return %[[MEMORY]]
