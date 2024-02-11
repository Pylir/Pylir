// RUN: pylir-opt %s | pylir-opt | FileCheck %s

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

// CHECK-LABEL: pyHIR.globalFunc @res_attr()
// CHECK-SAME: -> {test.name = 0 : i32}
pyHIR.globalFunc @res_attr() -> { test.name = 0 : i32 } {
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
  init_return %0
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
