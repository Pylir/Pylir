// RUN: pylir-opt %s | pylir-opt | FileCheck %s
// RUN: pylir-opt --verify-roundtrip

// CHECK-LABEL: pyHIR.globalFunc @test(
// CHECK-SAME: %{{[[:alnum:]]+}},
// CHECK-SAME: *%[[ARG1:[[:alnum:]]+]],
// CHECK-SAME: %{{.*}} "keyword",
// CHECK-SAME: %{{.*}} {test.name = 0 : i32},
// CHECK-SAME: %{{.*}} only "lol" has_default
// CHECK-NEXT: %{{.*}} = func "foo"(%{{.*}} "rest" = %[[ARG1]]) {

pyHIR.globalFunc @test(%arg0, *%arg1, %arg2 "keyword", %arg3 { test.name = 0 : i32 }, %arg4 only "lol" has_default) {
  %0 = func "foo"(%ff0 "rest" = %arg1) {
      return %ff0
  }
  return %0
}

// CHECK-LABEL: pyHIR.globalFunc @res_attr(%{{[[:alnum:]]+}})
// CHECK-SAME: -> {test.name = 0 : i32}
pyHIR.globalFunc @res_attr(%closure) -> { test.name = 0 : i32 } {
  %0 = func "foo"(%ff0 "rest") {
      return %ff0
  }
  return %0
}

// CHECK-LABEL: pyHIR.init "__main__" {
// CHECK: init_return
pyHIR.init "__main__" {
  %0 = func "foo"(%ff0 "rest") {
      return %ff0
  }
  init_return
}

// CHECK-LABEL: pyHIR.globalFunc @call(
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
pyHIR.globalFunc @call(%0) {
  // CHECK: call %[[ARG0]](%[[ARG0]], *%[[ARG0]], "rest"=%[[ARG0]], **%[[ARG0]])
  %1 = call %0(%0, *%0, "rest"=%0, **%0)
  // CHECK: call %[[ARG0]](%[[ARG0]])
  %2 = call %0(%0)
  // CHECK: call %[[ARG0]]()
  %3 = call %0()

  return %1
}

// CHECK-LABEL: pyHIR.globalFunc @callEx(
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
pyHIR.globalFunc @callEx(%0) {
  // CHECK: callEx %[[ARG0]]()
  // CHECK-NEXT: label ^{{.*}}(%[[ARG0]] : !py.dynamic) unwind ^{{.*}}(%[[ARG0]] : !py.dynamic)
  %2 = callEx %0()
    label ^bb1(%0 : !py.dynamic) unwind ^bb2(%0 : !py.dynamic)

^bb1(%arg0 : !py.dynamic):
  return %0

^bb2(%e : !py.dynamic, %arg1 : !py.dynamic):
  return %arg1
}

// CHECK-LABEL: pyHIR.globalFunc @binOp(
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
pyHIR.globalFunc @binOp(%0, %1) {
  // CHECK: binOp %[[ARG0]] __eq__ %[[ARG1]]
  %2 = binOp %0 __eq__ %1
  // CHECK: binOp %[[ARG0]] __ne__ %[[ARG1]]
  %3 = binOp %0 __ne__ %1
  // CHECK: binOp %[[ARG0]] __lt__ %[[ARG1]]
  %4 = binOp %0 __lt__ %1
  // CHECK: binOp %[[ARG0]] __le__ %[[ARG1]]
  %5 = binOp %0 __le__ %1
  // CHECK: binOp %[[ARG0]] __gt__ %[[ARG1]]
  %6 = binOp %0 __gt__ %1
  // CHECK: binOp %[[ARG0]] __ge__ %[[ARG1]]
  %7 = binOp %0 __ge__ %1
  // CHECK: binOp %[[ARG0]] __add__ %[[ARG1]]
  %8 = binOp %0 __add__ %1
  // CHECK: binOp %[[ARG0]] __sub__ %[[ARG1]]
  %9 = binOp %0 __sub__ %1
  // CHECK: binOp %[[ARG0]] __or__ %[[ARG1]]
  %10 = binOp %0 __or__ %1
  // CHECK: binOp %[[ARG0]] __xor__ %[[ARG1]]
  %11 = binOp %0 __xor__ %1
  // CHECK: binOp %[[ARG0]] __and__ %[[ARG1]]
  %12 = binOp %0 __and__ %1
  // CHECK: binOp %[[ARG0]] __lshift__ %[[ARG1]]
  %13 = binOp %0 __lshift__ %1
  // CHECK: binOp %[[ARG0]] __rshift__ %[[ARG1]]
  %14 = binOp %0 __rshift__ %1
  // CHECK: binOp %[[ARG0]] __mul__ %[[ARG1]]
  %15 = binOp %0 __mul__ %1
  // CHECK: binOp %[[ARG0]] __div__ %[[ARG1]]
  %16 = binOp %0 __div__ %1
  // CHECK: binOp %[[ARG0]] __floordiv__ %[[ARG1]]
  %17 = binOp %0 __floordiv__ %1
  // CHECK: binOp %[[ARG0]] __mod__ %[[ARG1]]
  %18 = binOp %0 __mod__ %1
  // CHECK: binOp %[[ARG0]] __matmul__ %[[ARG1]]
  %19 = binOp %0 __matmul__ %1
  return %2
}

// CHECK-LABEL: pyHIR.globalFunc @binExOp(
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
pyHIR.globalFunc @binExOp(%0, %1) {
  // CHECK: binOpEx %[[ARG0]] __eq__ %[[ARG1]]
  // CHECK-NEXT: label ^[[BB1:.*]](%[[ARG0]] : !py.dynamic) unwind ^[[BB2:.*]](%[[ARG1]] : !py.dynamic)
  %2 = binOpEx %0 __eq__ %1
    label ^bb1(%0 : !py.dynamic) unwind ^bb2(%1 : !py.dynamic)

^bb1(%arg0 : !py.dynamic):
  return %0

^bb2(%e : !py.dynamic, %arg1 : !py.dynamic):
  return %arg1
}

// CHECK-LABEL: pyHIR.globalFunc @binAssignOp(
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
pyHIR.globalFunc @binAssignOp(%0, %1) {
  // CHECK: binAssignOp %[[ARG0]] __iadd__ %[[ARG1]]
  %8 = binAssignOp %0 __iadd__ %1
  // CHECK: binAssignOp %[[ARG0]] __isub__ %[[ARG1]]
  %9 = binAssignOp %0 __isub__ %1
  // CHECK: binAssignOp %[[ARG0]] __ior__ %[[ARG1]]
  %10 = binAssignOp %0 __ior__ %1
  // CHECK: binAssignOp %[[ARG0]] __ixor__ %[[ARG1]]
  %11 = binAssignOp %0 __ixor__ %1
  // CHECK: binAssignOp %[[ARG0]] __iand__ %[[ARG1]]
  %12 = binAssignOp %0 __iand__ %1
  // CHECK: binAssignOp %[[ARG0]] __ilshift__ %[[ARG1]]
  %13 = binAssignOp %0 __ilshift__ %1
  // CHECK: binAssignOp %[[ARG0]] __irshift__ %[[ARG1]]
  %14 = binAssignOp %0 __irshift__ %1
  // CHECK: binAssignOp %[[ARG0]] __imul__ %[[ARG1]]
  %15 = binAssignOp %0 __imul__ %1
  // CHECK: binAssignOp %[[ARG0]] __idiv__ %[[ARG1]]
  %16 = binAssignOp %0 __idiv__ %1
  // CHECK: binAssignOp %[[ARG0]] __ifloordiv__ %[[ARG1]]
  %17 = binAssignOp %0 __ifloordiv__ %1
  // CHECK: binAssignOp %[[ARG0]] __imod__ %[[ARG1]]
  %18 = binAssignOp %0 __imod__ %1
  // CHECK: binAssignOp %[[ARG0]] __imatmul__ %[[ARG1]]
  %19 = binAssignOp %0 __imatmul__ %1
  return %19
}

pyHIR.init "foo" {
  %0 = py.constant(#py.dict<{}>)
  init_return
}

// CHECK-LABEL: pyHIR.globalFunc @initModule(
pyHIR.globalFunc @initModule(%closure) {
  %0 = py.constant(#py.dict<{}>)
  // CHECK: initModule @foo
  initModule @foo
  return %0
}

// CHECK-LABEL: pyHIR.globalFunc @subscription(
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
pyHIR.globalFunc @subscription(%0, %1) {
  // CHECK: %[[ITEM:.*]] = getItem %[[ARG0]][%[[ARG1]]]
  %2 = getItem %0[%1]
  // CHECK: setItem %[[ARG0]][%[[ITEM]]] to %[[ARG1]]
  setItem %0[%2] to %1
  // CHECK: delItem %[[ARG1]][%[[ARG0]]]
  delItem %1[%0]
  return %2
}

// CHECK-LABEL: pyHIR.globalFunc @contains(
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
pyHIR.globalFunc @contains(%container, %item) {
  // CHECK: contains %[[ARG1]] in %[[ARG0]]
  %0 = contains %item in %container
  return %0
}

// CHECK-LABEL: pyHIR.globalFunc @class(
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
pyHIR.globalFunc @class(%container, %item) {
  // CHECK: class "test"(%[[ARG1]], "metaclass"=%[[ARG0]])
  %0 = class "test"(%item, "metaclass"=%container) {
  ^bb0(%dict : !py.dynamic):
    class_return
  }
  // CHECK: classEx "test"(%[[ARG1]], "metaclass"=%[[ARG0]]) {
  // CHECK: } label ^{{[[:alnum:]]+}} unwind ^{{[[:alnum:]]+}}
  %1 = classEx "test"(%item, "metaclass"=%container) {
  ^bb0(%dict : !py.dynamic):
    class_return
  } label ^bb0 unwind ^bb1
^bb0:
  return %0
^bb1(%e : !py.dynamic):
  return %e
}
