// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.func private @test(%arg0 : !py.dynamic, %arg1 : !py.dynamic, %arg2 : !py.dynamic) -> !py.dynamic {
  return %arg0 : !py.dynamic
}

// CHECK-LABEL: func @make_function
// CHECK-SAME: %[[MEM:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
py.func @make_function(%mem: !pyMem.memory, %arg0: i32, %arg1: !py.dynamic) -> !py.dynamic {
  // CHECK: %[[FUNC:.*]] = llvm.mlir.addressof @test
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[MEM]][0, 1]
  // CHECK: store %[[FUNC]], %[[GEP]]
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[MEM]][0, 3]
  // CHECK-SAME: struct<{{.*}}, ({{.*}}, {{.*}}, {{.*}}, i32, ptr<{{[0-9]+}}>)>
  // CHECK: store %[[ARG0]], %[[GEP]]
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[MEM]][0, 4]
  // CHECK-SAME: struct<{{.*}}, ({{.*}}, {{.*}}, {{.*}}, i32, ptr<{{[0-9]+}}>)>
  // CHECK: store %[[ARG1]], %[[GEP]]
  %1 = pyMem.initFunc %mem to @test[%arg0, %arg1 : i32, !py.dynamic]
  return %1 : !py.dynamic
}
