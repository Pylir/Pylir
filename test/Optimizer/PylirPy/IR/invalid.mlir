// RUN: pylir-opt %s -split-input-file -verify-diagnostics

py.return // expected-error {{expected inside of a function op}}

// -----

func @test() -> i32 {
	py.return // expected-error {{'py.return' op operands are not compatible with enclosed function 'test's return types}}
}

// -----

func @test() -> i32 {
	%0 = arith.constant true
	py.return %0 : i1 // expected-error {{'py.return' op operands are not compatible with enclosed function 'test's return types}}
}
