// RUN: pylir-opt %s -convert-arith-to-llvm -convert-pylir-to-llvm --reconcile-unrealized-casts --split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str

py.func @foo() -> !pyMem.memory {
    %0 = constant(#builtins_str)
    %c0 = arith.constant 0 : index
    %1 = pyMem.gcAllocObject %0[%c0]
    return %1 : !pyMem.memory
}

// CHECK-DAG: #[[DESC:.*]] = #llvm.tbaa_type_desc<id = "Python Type Object"{{.*}}>
// CHECK-DAG: #[[$PYTHON_TYPE_OBJECT:.*]] = #llvm.tbaa_tag<base_type = #[[DESC]], access_type = #[[DESC]], offset = 0>

// CHECK-LABEL: llvm.func @foo
// CHECK-NEXT: %[[STR:.*]] = llvm.mlir.addressof @builtins.str
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : index)
// CHECK-NEXT: %[[INSTANCE_SIZE:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[POINTER_SIZE:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[MUL:.*]] = llvm.mul %[[ZERO]], %[[POINTER_SIZE]]
// CHECK-NEXT: %[[BYTES:.*]] = llvm.add %[[MUL]], %[[INSTANCE_SIZE]]
// CHECK-NEXT: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc(%[[BYTES]])
// CHECK-NEXT: %[[ZERO_I8:.*]] = llvm.mlir.constant(0 : i8)
// CHECK-NEXT: "llvm.intr.memset"(%[[MEMORY]], %[[ZERO_I8]], %[[BYTES]])
// CHECK-SAME: isVolatile = false
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][0, 0]
// CHECK-NEXT: llvm.store %[[STR]], %[[GEP]] {tbaa = [#[[$PYTHON_TYPE_OBJECT]]]}
// CHECK-NEXT: llvm.return %[[MEMORY]]

// CHECK: llvm.func @pylir_gc_alloc
// CHECK-NOT: llvm.noalias
// CHECK-SAME: attributes

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str

py.func @foo(%arg0 : !py.dynamic) -> !pyMem.memory {
    %c0 = arith.constant 0 : index
    %0 = pyMem.gcAllocObject %arg0[%c0]
    return %0 : !pyMem.memory
}

// CHECK-LABEL: llvm.func @foo
