// RUN: pylir-opt %s --pylir-trial-inliner='min-callee-size-reduction=0' --split-input-file | FileCheck %s

func @do_call(%arg0 : () -> i32) -> i32 {
	%0 = call_indirect %arg0() : () -> i32
	return %0 : i32
}

func @foo() -> i32 {
	%0 = arith.constant 5 : i32
	return %0 : i32
}

func @test() -> i32 {
    %0 = func.constant @foo : () -> i32
    %1 = call @do_call(%0) : (() -> i32) -> i32
    return %1 : i32
}

// CHECK-LABEL: @test
// CHECK: %[[C:.*]] = arith.constant 5
// CHECK: return %[[C]]
