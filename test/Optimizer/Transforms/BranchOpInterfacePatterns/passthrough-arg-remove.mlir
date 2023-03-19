// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.func private @foo() -> i32

py.func @test(%arg1 : i32) -> i32 {
	%0 = invoke @foo() : () -> i32
		label ^bb1 unwind ^bb2

^bb1:
	cf.br ^bb3(%0 : i32)

^bb2(%e : !py.dynamic):
	test.use(%e) : !py.dynamic
	%1 = arith.constant 5 : i32
	cf.br ^bb3(%1 : i32)

^bb3(%b0: i32):
	return %b0 : i32
}

// CHECK-LABEL: @test
// CHECK: %[[VALUE:.*]] = invoke
// CHECK-NEXT: label ^[[BB1:[[:alnum:]]+]]
// CHECK-NEXT: ^[[BB1]]:
// CHECK-NEXT: cf.br ^{{.*}}(%[[VALUE]] : i32)

// -----

py.func private @foo() -> i32

py.func @test(%arg1 : i32) {
	%0 = invoke @foo() : () -> i32
		label ^bb1 unwind ^bb2

^bb1:
	cf.br ^bb3

^bb2(%e : !py.dynamic):
	cf.br ^bb3

^bb3:
	return
}

// CHECK-LABEL: @test
// CHECK: %[[VALUE:.*]] = invoke
// CHECK-NEXT: label ^[[BB3:[[:alnum:]]+]] unwind ^[[BB2:[[:alnum:]]+]]
// CHECK-NEXT: ^[[BB2]](%{{.*}}: !py.dynamic):
// CHECK-NEXT: cf.br ^[[BB3]]
// CHECK-NEXT: ^[[BB3]]:
// CHECK-NEXT: return
