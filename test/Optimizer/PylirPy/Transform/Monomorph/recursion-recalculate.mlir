// RUN: pylir-opt %s --pylir-monomorph --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

py.globalValue "private" @test

func.func @foo(%arg0 : !py.dynamic) -> !py.dynamic {
	%0 = test.random
	cf.cond_br %0, ^call, ^exit(%arg0 : !py.dynamic)

^call:
	%1 = py.constant(#py.int<1>)
	%2 = py.int.add %arg0, %1
	%3 = py.call @foo(%2) : (!py.dynamic) -> !py.dynamic
	%4 = py.constant(@test)
	return %4 : !py.dynamic

^exit(%5 : !py.dynamic):
	%6 = py.constant(#py.int<2>)
    %7 = py.int.add %5, %6
	return %7 : !py.dynamic
}

func.func @__init__() -> !py.dynamic {
	%1 = py.constant(#py.int<0>)
	%2 = py.call @foo(%1) : (!py.dynamic) -> !py.dynamic
	%3 = py.typeOf %2
	return %3 : !py.dynamic
}

// CHECK-LABEL: func @__init__
// CHECK: %[[TYPE:.*]] = py.typeOf
// CHECK: return %[[TYPE]]
