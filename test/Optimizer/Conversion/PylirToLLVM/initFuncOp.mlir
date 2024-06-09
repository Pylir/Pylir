// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.func private @test(%arg0 : !py.dynamic, %arg1 : !py.dynamic, %arg2 : !py.dynamic) -> !py.dynamic {
  return %arg0 : !py.dynamic
}

// CHECK-LABEL: func @make_function
// CHECK-SAME: %[[MEM:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
py.func @make_function(%mem: !pyMem.memory, %arg0: !py.dynamic, %arg1: i32) -> !py.dynamic {
  // CHECK: %[[FUNC:.*]] = llvm.mlir.addressof @test
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[MEM]][0, 1]
  // CHECK: store %[[FUNC]], %[[GEP]]
  // CHECK: %[[CLOSURE_SIZE:.*]] = llvm.mlir.constant({{(12|8)}} : i32)
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[MEM]][0, 2]
  // CHECK: store %[[CLOSURE_SIZE]], %[[GEP]]
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[MEM]][0, 4]
  // CHECK-SAME: struct<{{.*}}, ({{.*}}, {{.*}}, {{.*}}, {{.*}}, ptr<{{[0-9]+}}>, i32, array<1 x i8>)>
  // CHECK: store %[[ARG0]], %[[GEP]]
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[MEM]][0, 5]
  // CHECK-SAME: struct<{{.*}}, ({{.*}}, {{.*}}, {{.*}}, {{.*}}, ptr<{{[0-9]+}}>, i32, array<1 x i8>)>
  // CHECK: store %[[ARG1]], %[[GEP]]
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[MEM]][0, 6]
  // CHECK: %[[MASK:.*]] = llvm.mlir.constant(1 : i8)
  // CHECK: %[[GEP2:.*]] = llvm.getelementptr %[[GEP]][0, 0]
  // CHECK: store %[[MASK]], %[[GEP2]]
  %1 = pyMem.initFunc %mem to @test[%arg0, %arg1 : !py.dynamic, i32]
  return %1 : !py.dynamic
}
