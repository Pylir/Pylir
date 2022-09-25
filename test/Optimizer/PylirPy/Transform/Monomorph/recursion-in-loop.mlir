// RUN: pylir-opt %s --pylir-monomorph --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

py.globalValue "private" @test

func.func @foo(%arg0 : !py.dynamic) -> !py.dynamic {
	%0 = test.random
	cf.cond_br %0, ^call, ^exit

^call:
    %1 = test.random
    cf.cond_br %1, ^body, ^loopExit

^body:
	%2 = py.constant(#py.int<1>)
	%3 = py.int.add %arg0, %2
	%4 = py.call @foo(%3) : (!py.dynamic) -> !py.dynamic
	cf.br ^call

^loopExit:
	%5 = py.constant(#py.ref<@test>)
	return %5 : !py.dynamic

^exit:
	%6 = py.constant(#py.int<2>)
    %7 = py.int.add %arg0, %6
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
