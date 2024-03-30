// RUN: pylir-opt %s --convert-pylirHIR-to-pylirPy --split-input-file | FileCheck %s

// CHECK-LABEL: func @test$impl
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
pyHIR.globalFunc @test(%0, %1) {
  // CHECK: %[[ITEM:.*]] = call @pylir__contains__(%[[ARG0]], %[[ARG1]])
  %2 = contains %1 in %0
  return %2
}
