// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

// CHECK-LABEL: func @foo
py.func @foo(%arg0 : i32, %arg1 : !py.dynamic) -> !pyMem.memory {
  // CHECK: %[[NULL:.*]] = llvm.mlir.zero
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[NULL]][1]
  // CHECK-SAME: !llvm.struct<"{{.*}}", (ptr<{{[0-9]+}}>, ptr, i32, array<{{[0-9]+}} x ptr<{{[0-9]+}}>>, i32, ptr<{{[0-9]+}}>, array<1 x i8>)>
  // CHECK: %[[SIZE:.*]] = llvm.ptrtoint %[[GEP]]
  // CHECK: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc(%[[SIZE]])
  // CHECK: "llvm.intr.memset"(%[[MEMORY]], %{{.*}}, %[[SIZE]])
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][0, 0]
  // CHECK: %[[TYPE:.*]] = llvm.mlir.addressof @builtins.function
  // CHECK: llvm.store %[[TYPE]], %[[GEP]]
  %0 = pyMem.gcAllocFunction [i32, !py.dynamic]
  return %0 : !pyMem.memory
}
