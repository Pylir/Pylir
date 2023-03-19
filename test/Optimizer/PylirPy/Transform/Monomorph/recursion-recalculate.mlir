// RUN: pylir-opt %s --pylir-monomorph --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.int = #py.type

py.globalValue "private" @test

py.func @foo(%arg0 : !py.dynamic) -> !py.dynamic {
	%0 = test.random
	cf.cond_br %0, ^call, ^exit(%arg0 : !py.dynamic)

^call:
	%1 = constant(#py.int<1>)
	%2 = py.int.add %arg0, %1
	%3 = call @foo(%2) : (!py.dynamic) -> !py.dynamic
	%4 = constant(#py.ref<@test>)
	return %4 : !py.dynamic

^exit(%5 : !py.dynamic):
	%6 = constant(#py.int<2>)
    %7 = py.int.add %5, %6
	return %7 : !py.dynamic
}

py.func @__init__() -> !py.dynamic {
	%1 = constant(#py.int<0>)
	%2 = call @foo(%1) : (!py.dynamic) -> !py.dynamic
	%3 = typeOf %2
	return %3 : !py.dynamic
}

// CHECK-LABEL: func @__init__
// CHECK: %[[TYPE:.*]] = typeOf
// CHECK: return %[[TYPE]]
