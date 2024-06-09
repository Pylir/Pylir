// RUN: pylir-opt %s -convert-pylir-to-llvm | FileCheck %s

// CHECK-LABEL: func @func_get_closure
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
py.func @func_get_closure(%arg0 : !py.dynamic) -> i32 {
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ARG0]][0, 6]
  // CHECK-SAME: struct<{{.*}}, ({{.*}}, {{.*}}, {{.*}}, {{.*}}, i32, i64, i32{{.*}})>
  // CHECK: %[[LOADED:.*]] = llvm.load %[[GEP]]
  // CHECK-SAME: -> i32
  %0 = function_closureArg %arg0[2] : [i32, i64, i32]
  // CHECK: return %[[LOADED]]
  return %0 : i32
}
