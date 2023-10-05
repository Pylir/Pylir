// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.global @foo : !py.dynamic

py.func @test() -> !py.dynamic {
  %0 = constant(#py.str<"value">)
  store %0 : !py.dynamic into @foo
  cf.br ^bb1(%0 : !py.dynamic)

^bb1(%1: !py.dynamic):
  %2 = test.random
  cf.cond_br %2, ^bb1(%1 : !py.dynamic), ^bb2

^bb2:
  return %1 : !py.dynamic
}

// CHECK-LABEL: py.func @test
// CHECK: %[[C:.*]] = constant(#py.str<"value">)
// CHECK: cf.br ^[[BB1:[[:alnum:]]+]]
// CHECK-NOT: (

// CHECK: ^[[BB1]]:
// CHECK: cf.cond_br %{{.*}}, ^[[BB1]], ^[[BB2:[[:alnum:]]+]]

// CHECK: ^[[BB2]]:
// CHECK: return %[[C]]

// -----

py.func private @foo()

py.func @test(%arg1 : i32) -> i32 {
	py.invoke @foo() : () -> ()
		label ^bb2 unwind ^bb1(%arg1 : i32)

^bb1(%e : !py.dynamic, %b0 : i32):
	return %b0 : i32

^bb2:
	test.use(%arg1) : i32
	py.unreachable
}

// CHECK-LABEL: @test
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-NEXT: invoke
// CHECK-NEXT: label ^{{.*}} unwind ^[[BB1:[[:alnum:]]+]]
// CHECK-NEXT: ^[[BB1]]
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-NOT: %{{[[:alnum:]]+}}
// CHECK-NEXT: return %[[ARG1]]

// -----

py.func private @foo()

py.func @test(%arg1 : i32, %arg2 : i32) -> (i32, i32) {
	py.invoke @foo() : () -> ()
		label ^bb2 unwind ^bb1(%arg1, %arg2 : i32, i32)

^bb1(%e : !py.dynamic, %b0 : i32, %b1 : i32):
	return %b0, %b1 : i32,  i32

^bb2:
	test.use(%arg1) : i32
	test.use(%arg2) : i32
	py.unreachable
}

// CHECK-LABEL: @test
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG2:[[:alnum:]]+]]
// CHECK-NEXT: invoke
// CHECK-NEXT: label ^{{.*}} unwind ^[[BB1:[[:alnum:]]+]]
// CHECK-NEXT: ^[[BB1]]
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-NOT: %{{[[:alnum:]]+}}
// CHECK-NEXT: return %[[ARG1]], %[[ARG2]]
