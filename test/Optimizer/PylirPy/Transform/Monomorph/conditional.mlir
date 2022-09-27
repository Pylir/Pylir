// RUN: pylir-opt %s --pylir-monomorph --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.int = #py.type

py.globalValue "private" @test

func.func @foo(%arg0 : !py.dynamic) -> !py.dynamic {
	%0 = py.typeOf %arg0
	%1 = py.constant(#py.ref<@builtins.int>)
	%2 = py.is %0, %1
	cf.cond_br %2, ^exit(%arg0 : !py.dynamic), ^call

^call:
	%3 = py.constant(#py.ref<@test>)
	cf.br ^exit(%3 : !py.dynamic)

^exit(%4 : !py.dynamic):
	return %4 : !py.dynamic
}

func.func @__init__() -> !py.dynamic {
	%1 = py.constant(#py.int<0>)
	%2 = py.call @foo(%1) : (!py.dynamic) -> !py.dynamic
	%3 = py.typeOf %2
	return %3 : !py.dynamic
}

// CHECK-LABEL: func @__init__
// CHECK: %[[TYPE:.*]] = py.constant(#py.ref<@builtins.int>)
// CHECK: return %[[TYPE]]
