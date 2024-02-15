// RUN: pylir-opt %s --convert-pylirHIR-to-pylirPy --split-input-file | FileCheck %s

// CHECK-LABEL: func @binOp$impl
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
pyHIR.globalFunc @binOp(%0, %1) {
  // CHECK: call @pylir__eq__(%[[ARG0]], %[[ARG1]])
  %2 = binOp %0 __eq__ %1
  // CHECK: call @pylir__ne__(%[[ARG0]], %[[ARG1]])
  %3 = binOp %0 __ne__ %1
  // CHECK: call @pylir__lt__(%[[ARG0]], %[[ARG1]])
  %4 = binOp %0 __lt__ %1
  // CHECK: call @pylir__le__(%[[ARG0]], %[[ARG1]])
  %5 = binOp %0 __le__ %1
  // CHECK: call @pylir__gt__(%[[ARG0]], %[[ARG1]])
  %6 = binOp %0 __gt__ %1
  // CHECK: call @pylir__ge__(%[[ARG0]], %[[ARG1]])
  %7 = binOp %0 __ge__ %1
  // CHECK: call @pylir__add__(%[[ARG0]], %[[ARG1]])
  %8 = binOp %0 __add__ %1
  // CHECK: call @pylir__sub__(%[[ARG0]], %[[ARG1]])
  %9 = binOp %0 __sub__ %1
  // CHECK: call @pylir__or__(%[[ARG0]], %[[ARG1]])
  %10 = binOp %0 __or__ %1
  // CHECK: call @pylir__xor__(%[[ARG0]], %[[ARG1]])
  %11 = binOp %0 __xor__ %1
  // CHECK: call @pylir__and__(%[[ARG0]], %[[ARG1]])
  %12 = binOp %0 __and__ %1
  // CHECK: call @pylir__lshift__(%[[ARG0]], %[[ARG1]])
  %13 = binOp %0 __lshift__ %1
  // CHECK: call @pylir__rshift__(%[[ARG0]], %[[ARG1]])
  %14 = binOp %0 __rshift__ %1
  // CHECK: call @pylir__mul__(%[[ARG0]], %[[ARG1]])
  %15 = binOp %0 __mul__ %1
  // CHECK: call @pylir__div__(%[[ARG0]], %[[ARG1]])
  %16 = binOp %0 __div__ %1
  // CHECK: call @pylir__floordiv__(%[[ARG0]], %[[ARG1]])
  %17 = binOp %0 __floordiv__ %1
  // CHECK: call @pylir__mod__(%[[ARG0]], %[[ARG1]])
  %18 = binOp %0 __mod__ %1
  // CHECK: call @pylir__matmul__(%[[ARG0]], %[[ARG1]])
  %19 = binOp %0 __matmul__ %1
  return %2
}

// CHECK-LABEL: func @binOpEx$impl
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
pyHIR.globalFunc @binOpEx(%0, %1) {
  // CHECK: invoke @pylir__eq__(%[[ARG0]], %[[ARG1]])
  // CHECK-NEXT: label ^[[BB1:.*]] unwind ^[[BB2:.*]](%[[ARG1]] : !py.dynamic)
  %2 = binOpEx %0 __eq__ %1
    label ^bb1(%0 : !py.dynamic) unwind ^bb2(%1 : !py.dynamic)

// CHECK: ^[[BB1]]:
// CHECK: cf.br ^{{.*}}(%[[ARG0]] : !py.dynamic)
^bb1(%arg0 : !py.dynamic):
  return %0

// CHECK: ^[[BB2]]({{.*}}):
// CHECK: call @pylir__eq__
// CHECK: return
^bb2(%e : !py.dynamic, %arg1 : !py.dynamic):
  %3 = binOp %e __eq__ %arg1
  return %3
}
