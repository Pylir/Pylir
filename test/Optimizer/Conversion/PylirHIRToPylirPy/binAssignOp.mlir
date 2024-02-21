// RUN: pylir-opt %s --convert-pylirHIR-to-pylirPy --split-input-file | FileCheck %s

// CHECK-LABEL: func @binAssignOp$impl
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
pyHIR.globalFunc @binAssignOp(%0, %1) {
  // CHECK: call @pylir__iadd__(%[[ARG0]], %[[ARG1]])
  %8 = binAssignOp %0 __iadd__ %1
  // CHECK: call @pylir__isub__(%[[ARG0]], %[[ARG1]])
  %9 = binAssignOp %0 __isub__ %1
  // CHECK: call @pylir__ior__(%[[ARG0]], %[[ARG1]])
  %10 = binAssignOp %0 __ior__ %1
  // CHECK: call @pylir__ixor__(%[[ARG0]], %[[ARG1]])
  %11 = binAssignOp %0 __ixor__ %1
  // CHECK: call @pylir__iand__(%[[ARG0]], %[[ARG1]])
  %12 = binAssignOp %0 __iand__ %1
  // CHECK: call @pylir__ilshift__(%[[ARG0]], %[[ARG1]])
  %13 = binAssignOp %0 __ilshift__ %1
  // CHECK: call @pylir__irshift__(%[[ARG0]], %[[ARG1]])
  %14 = binAssignOp %0 __irshift__ %1
  // CHECK: call @pylir__imul__(%[[ARG0]], %[[ARG1]])
  %15 = binAssignOp %0 __imul__ %1
  // CHECK: call @pylir__idiv__(%[[ARG0]], %[[ARG1]])
  %16 = binAssignOp %0 __idiv__ %1
  // CHECK: call @pylir__ifloordiv__(%[[ARG0]], %[[ARG1]])
  %17 = binAssignOp %0 __ifloordiv__ %1
  // CHECK: call @pylir__imod__(%[[ARG0]], %[[ARG1]])
  %18 = binAssignOp %0 __imod__ %1
  // CHECK: call @pylir__imatmul__(%[[ARG0]], %[[ARG1]])
  %19 = binAssignOp %0 __imatmul__ %1
  return %19
}
