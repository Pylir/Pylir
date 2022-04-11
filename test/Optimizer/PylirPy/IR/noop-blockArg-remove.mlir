// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

func @test(%arg0 : i1, %arg1 : i32) -> i32 {
	py.cond_br %arg0, ^bb1(%arg1 : i32), ^bb2

^bb1(%b0 : i32):
	py.return %b0 : i32

^bb2:
	test.use(%arg1) : i32
	py.unreachable
}

// CHECK-LABEL: @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-NEXT: py.cond_br %[[ARG0]], ^[[BB1:[[:alnum:]]+]]
// CHECK-NEXT: ^[[BB1]]:
// CHECK-NEXT: py.return %[[ARG1]]

// -----

func private @foo()

func @test(%arg1 : i32) -> i32 {
	py.invoke @foo() : () -> ()
		label ^bb2 unwind ^bb1(%arg1 : i32)

^bb1(%e : !py.unknown, %b0 : i32):
	py.return %b0 : i32

^bb2:
	test.use(%arg1) : i32
	py.unreachable
}

// CHECK-LABEL: @test
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-NEXT: py.invoke
// CHECK-NEXT: label ^{{.*}} unwind ^[[BB1:[[:alnum:]]+]]
// CHECK-NEXT: ^[[BB1]]
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-NOT: %{{[[:alnum:]]+}}
// CHECK-NEXT: py.return %[[ARG1]]
