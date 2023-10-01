// RUN: pylir-opt %s -convert-arith-to-llvm -convert-pylir-to-llvm --reconcile-unrealized-casts --split-input-file | FileCheck %s


#builtins_type = #py.globalValue<builtins.type, const, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str
#builtins_tuple = #py.globalValue<builtins.tuple, const, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple

py.func @foo(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = constant(#builtins_str)
    %c0 = arith.constant 0 : index
    %1 = pyMem.gcAllocObject %0[%c0]
    %2 = pyMem.initStr %1 to %arg0, %arg1
    return %2 : !py.dynamic
}

// CHECK-DAG: #[[DESC:.*]] = #llvm.tbaa_type_desc<id = "Python String Size"{{.*}}>
// CHECK-DAG: #[[$PYTHON_STRING_SIZE:.*]] = #llvm.tbaa_tag<base_type = #[[DESC]], access_type = #[[DESC]], offset = 0>
// CHECK-DAG: #[[DESC:.*]] = #llvm.tbaa_type_desc<id = "Python String Capacity"{{.*}}>
// CHECK-DAG: #[[$PYTHON_STRING_CAPACITY:.*]] = #llvm.tbaa_tag<base_type = #[[DESC]], access_type = #[[DESC]], offset = 0>
// CHECK-DAG: #[[DESC:.*]] = #llvm.tbaa_type_desc<id = "Python String Element Ptr"{{.*}}>
// CHECK-DAG: #[[$PYTHON_STRING_ELEMENT:.*]] = #llvm.tbaa_tag<base_type = #[[DESC]], access_type = #[[DESC]], offset = 0>


// CHECK-LABEL: @foo
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-NEXT: %[[STR:.*]] = llvm.mlir.addressof @builtins.str
// CHECK: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc
// CHECK: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][0, 0]
// CHECK-NEXT: llvm.store %[[STR]], %[[GEP]]
// CHECK-NEXT: %[[BUFFER:.*]] = llvm.getelementptr %[[MEMORY]][0, 1]
// CHECK-NEXT: %[[ZERO_I:.*]] = llvm.mlir.constant(0 : index)
// CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[ARG0]][0, 1]
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[GEP1]][0, 0]
// CHECK-NEXT: %[[SIZE_0:.*]] = llvm.load %[[GEP2]] {tbaa = [#[[$PYTHON_STRING_SIZE]]]}
// CHECK-NEXT: %[[SIZE_SUM_0:.*]] = llvm.add %[[SIZE_0]], %[[ZERO_I]]
// CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[ARG1]][0, 1]
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[GEP1]][0, 0]
// CHECK-NEXT: %[[SIZE_1:.*]] = llvm.load %[[GEP2]] {tbaa = [#[[$PYTHON_STRING_SIZE]]]}
// CHECK-NEXT: %[[SIZE:.*]] = llvm.add %[[SIZE_SUM_0]], %[[SIZE_1]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[BUFFER]][0, 0]
// CHECK-NEXT: llvm.store %[[SIZE]], %[[GEP]] {tbaa = [#[[$PYTHON_STRING_SIZE]]]}
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[BUFFER]][0, 1]
// CHECK-NEXT: llvm.store %[[SIZE]], %[[GEP]] {tbaa = [#[[$PYTHON_STRING_CAPACITY]]]}
// CHECK-NEXT: %[[ARRAY:.*]] = llvm.call @malloc(%[[SIZE]])
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[BUFFER]][0, 2]
// CHECK-NEXT: llvm.store %[[ARRAY]], %[[GEP]] {tbaa = [#[[$PYTHON_STRING_ELEMENT]]]}
// CHECK-NEXT: %[[SIZE:.*]] = llvm.mlir.constant(0 : index)

// CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[ARG0]][0, 1]
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[GEP1]][0, 0]
// CHECK-NEXT: %[[SIZE_0:.*]] = llvm.load %[[GEP2]] {tbaa = [#[[$PYTHON_STRING_SIZE]]]}
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[GEP1]][0, 2]
// CHECK-NEXT: %[[ARRAY_0:.*]] = llvm.load %[[GEP2]] {tbaa = [#[[$PYTHON_STRING_ELEMENT]]]}
// CHECK-NEXT: "llvm.intr.memcpy"(%[[ARRAY]], %[[ARRAY_0]], %[[SIZE_0]])
// CHECK-SAME: isVolatile = false
// CHECK-NEXT: %[[SIZE_NEW:.*]] = llvm.add %[[SIZE_0]], %[[SIZE]]

// CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[ARG1]][0, 1]
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[GEP1]][0, 0]
// CHECK-NEXT: %[[SIZE_1:.*]] = llvm.load %[[GEP2]] {tbaa = [#[[$PYTHON_STRING_SIZE]]]}
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[GEP1]][0, 2]
// CHECK-NEXT: %[[ARRAY_1:.*]] = llvm.load %[[GEP2]] {tbaa = [#[[$PYTHON_STRING_ELEMENT]]]}
// CHECK-NEXT: %[[DEST:.*]] = llvm.getelementptr %[[ARRAY]][%[[SIZE_NEW]]]
// CHECK-NEXT: "llvm.intr.memcpy"(%[[DEST]], %[[ARRAY_1]], %[[SIZE_1]])
// CHECK-SAME: isVolatile = false
// CHECK-NEXT: llvm.add %[[SIZE_NEW]], %[[SIZE_1]]

// CHECK-NEXT: llvm.return %[[MEMORY]]
