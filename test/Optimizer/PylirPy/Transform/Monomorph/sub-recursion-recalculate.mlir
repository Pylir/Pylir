// RUN: pylir-opt %s --pylir-monomorph --canonicalize --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.int = #py.type

py.globalValue "private" @test

func.func @plusOne(%arg0 : !py.dynamic) -> !py.dynamic {
	%0 = py.call @header(%arg0) : (!py.dynamic) -> !py.dynamic
	return %0 : !py.dynamic
}

func.func @plusTwo(%arg0 : !py.dynamic) -> !py.dynamic {
	%0 = test.random
	cf.cond_br %0, ^bb1, ^bb2

^bb1:
	%1 = py.call @plusOne(%arg0) : (!py.dynamic) -> !py.dynamic
	%2 = py.constant(@test)
    return %2 : !py.dynamic

^bb2:
	return %arg0 : !py.dynamic
}

func.func @header(%arg0 : !py.dynamic) -> !py.dynamic {
	%0 = test.random
	cf.cond_br %0, ^bb1, ^bb2

^bb1:
	%1 = py.call @plusOne(%arg0) : (!py.dynamic) -> !py.dynamic
	%3 = py.typeOf %1
	test.use(%3) : !py.dynamic
	return %1 : !py.dynamic

^bb2:
	%2 = py.call @plusTwo(%arg0) : (!py.dynamic) -> !py.dynamic
    return %2 : !py.dynamic
}

func.func @root() -> !py.dynamic {
	%0 = py.constant(#py.int<1>)
	%1 = py.call @header(%0) : (!py.dynamic) -> !py.dynamic
	%2 = py.typeOf %1
	return %2 : !py.dynamic
}

// CHECK-LABEL: func.func @root
// CHECK-NOT: py.constant(@builtins.int)
