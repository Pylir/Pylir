// RUN: pylir-opt %s --convert-pylirHIR-to-pylirPy --split-input-file | FileCheck %s

// CHECK-LABEL: func @test$impl
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
pyHIR.globalFunc @test(%0, %1) {
  // CHECK: %[[STR:.*]] = constant(#py.str<"test">)
  // CHECK: call @pylir__setattr__(%[[ARG0]], %[[STR]], %[[ARG1]])
  setAttr "test" of %0 to %1
  return %0
}
