// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

// CHECK-LABEL: func @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
py.func @test(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
  // CHECK: br ^[[BB:.*]](%[[ARG0]], %[[ARG1]] : !{{.*}}, !{{.*}})
  raiseEx %arg0
    label ^bb1 unwind ^bb2(%arg1 : !py.dynamic)
^bb1:
  unreachable

// CHECK: ^[[BB]](%[[E:.*]]: !{{.*}}, %{{.*}}: !{{.*}}):
^bb2(%e: !py.dynamic, %arg2 : !py.dynamic):
  // CHECK: return %[[E]]
  return %e : !py.dynamic
}
