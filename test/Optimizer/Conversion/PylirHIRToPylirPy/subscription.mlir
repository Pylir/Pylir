// RUN: pylir-opt %s --convert-pylirHIR-to-pylirPy --split-input-file | FileCheck %s

// CHECK-LABEL: func @test$impl
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
pyHIR.globalFunc @test(%0, %1) {
  // CHECK: %[[ITEM:.*]] = call @pylir__getitem__(%[[ARG0]], %[[ARG1]])
  %2 = getItem %0[%1]
  // CHECK: call @pylir__setitem__(%[[ARG0]], %[[ITEM]], %[[ARG1]])
  setItem %0[%2] to %1
  // CHECK: call @pylir__delitem__(%[[ARG1]], %[[ARG0]])
  delItem %1[%0]
  return %2
}
