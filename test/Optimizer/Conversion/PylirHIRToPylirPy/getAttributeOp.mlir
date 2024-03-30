// RUN: pylir-opt %s --convert-pylirHIR-to-pylirPy --split-input-file | FileCheck %s

// CHECK-LABEL: func @test$impl
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
pyHIR.globalFunc @test(%0) {
  // CHECK: %[[STR:.*]] = constant(#py.str<"test">)
  // CHECK: %[[ITEM:.*]] = call @pylir__getattribute__(%[[ARG0]], %[[STR]])
  %2 = getAttribute "test" of %0
  return %2
}
