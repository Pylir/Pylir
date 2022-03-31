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

// -----

func @test() {
	py.call @foo() : () -> () // expected-error {{'py.call' op failed to find function named 'foo'}}
}

// -----

func @test() {
	%0 = arith.constant true
	py.call @test(%0) : (i1) -> () // expected-error {{call operand types are not compatible with argument types of 'test'}}
}
