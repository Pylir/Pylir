// RUN: pylir-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: @foo
py.func @foo() {
  // CHECK-NEXT: constant
  // CHECK-SAME: {
  // CHECK-SAME: #py.int<5> to #py.int<8>
  // CHECK-NOT: ,
  // CHECK-SAME: }
  %0 = constant(#py.dict<{#py.int<5> to #py.int<3>, #py.float<5.0> to #py.int<8>}>)
  return
}
