// RUN: pylir-opt %s --pylir-inliner --split-input-file | FileCheck %s

py.func private @complex(i32) -> i32

py.func @bar() -> i32 {
	%0 = arith.constant 6 : i32
	%1 = call @complex(%0) : (i32) -> i32
	return %1 : i32
}

py.func @foo() -> i1 {
	%0 = arith.constant true
	return %0 : i1
}

py.func @test() -> i32 {
    %0 = call @foo() : () -> i1
    cf.cond_br %0, ^bb0, ^bb1

^bb0:
	%1 = arith.constant 5 : i32
    return %1 : i32

^bb1:
	%2 = call @bar() : () -> i32
    return %2 : i32
}

// CHECK-LABEL: @test
// CHECK: %[[C:.*]] = arith.constant 5
// CHECK: return %[[C]]
