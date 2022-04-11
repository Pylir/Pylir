// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

func @test(%arg0 : i1, %arg1 : i32) -> i32 {
	py.cond_br %arg0, ^bb1, ^bb2

^bb1:
	py.br ^bb3(%arg1 : i32)

^bb2:
	test.use(%arg1) : i32
	%0 = arith.constant 5 : i32
	py.br ^bb3(%0 : i32)

^bb3(%b0: i32):
	py.return %b0 : i32
}

// CHECK-LABEL: @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK: py.cond_br %[[ARG0]], ^[[BB3:[[:alnum:]]+]](%[[ARG1]] : i32)
// CHECK: ^[[BB3]](%[[B0:[[:alnum:]]+]]: i32):
// CHECK-NEXT: py.return %[[B0]]

// -----

func private @foo() -> i32

func @test(%arg1 : i32) -> i32 {
	%0 = py.invoke @foo() : () -> i32
		label ^bb1 unwind ^bb2

^bb1:
	py.br ^bb3(%0 : i32)

^bb2(%e : !py.unknown):
	test.use(%e) : !py.unknown
	%1 = arith.constant 5 : i32
	py.br ^bb3(%1 : i32)

^bb3(%b0: i32):
	py.return %b0 : i32
}

// CHECK-LABEL: @test
// CHECK: %[[VALUE:.*]] = py.invoke
// CHECK-NEXT: label ^[[BB1:[[:alnum:]]+]]
// CHECK-NEXT: ^[[BB1]]:
// CHECK-NEXT: py.br ^{{.*}}(%[[VALUE]] : i32)
