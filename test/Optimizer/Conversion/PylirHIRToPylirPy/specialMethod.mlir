// RUN: pylir-opt %s --convert-pylirHIR-to-pylirPy --split-input-file | FileCheck %s

// CHECK-LABEL: func @specialMethod$impl(
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG2:[[:alnum:]]+]]
pyHIR.globalFunc @specialMethod(%arg0, %arg1, %arg2) {
  // CHECK: call @pylir__getitem__(%[[ARG0]], %[[ARG1]])
  specialMethod %arg0 __getitem__(%arg1)
  // CHECK: call @pylir__setitem__(%[[ARG0]], %[[ARG1]], %[[ARG2]])
  specialMethod %arg0 __setitem__(%arg1, %arg2)
  // CHECK: call @pylir__delitem__(%[[ARG0]], %[[ARG1]])
  specialMethod %arg0 __delitem__(%arg1)
  // CHECK: call @pylir__neg__(%[[ARG0]])
  specialMethod %arg0 __neg__()
  // CHECK: call @pylir__pos__(%[[ARG0]])
  specialMethod %arg0 __pos__()
  // CHECK: call @pylir__invert__(%[[ARG0]])
  specialMethod %arg0 __invert__()
  // CHECK: call @pylir__getattr__(%[[ARG0]], %[[ARG1]])
  specialMethod %arg0 __getattr__(%arg1)
  // CHECK: call @pylir__setattr__(%[[ARG0]], %[[ARG1]], %[[ARG2]])
  specialMethod %arg0 __setattr__(%arg1, %arg2)
  return %arg0
}
