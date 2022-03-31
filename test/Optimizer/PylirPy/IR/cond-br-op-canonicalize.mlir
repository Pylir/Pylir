// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

func @test(%arg0 : i32) -> i32 {
	%0 = arith.constant true
	py.cond_br %0, ^true(%arg0 : i32), ^false

^true(%1 : i32):
	return %1 : i32

^false:
	%2 = arith.constant 0 : i32
	return %2 : i32
}

// CHECK-LABEL: func @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: return %[[ARG0]]
