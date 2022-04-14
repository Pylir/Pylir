// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

func private @foo()

func @test(%arg1 : i32) -> i32 {
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
// CHECK-NEXT: py.invoke
// CHECK-NEXT: label ^{{.*}} unwind ^[[BB1:[[:alnum:]]+]]
// CHECK-NEXT: ^[[BB1]]
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-NOT: %{{[[:alnum:]]+}}
// CHECK-NEXT: return %[[ARG1]]
