// RUN: pylir-opt %s --pylir-monomorph --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.int = #py.type

py.globalValue "private" @test

py.func @foo(%arg0 : !py.dynamic) -> !py.dynamic {
	%0 = typeOf %arg0
	%1 = constant(#py.ref<@builtins.int>)
	%2 = is %0, %1
	cf.cond_br %2, ^exit(%arg0 : !py.dynamic), ^call

^call:
	%3 = constant(#py.ref<@test>)
	cf.br ^exit(%3 : !py.dynamic)

^exit(%4 : !py.dynamic):
	return %4 : !py.dynamic
}

py.func @__init__() -> !py.dynamic {
	%1 = constant(#py.int<0>)
	%2 = call @foo(%1) : (!py.dynamic) -> !py.dynamic
	%3 = typeOf %2
	return %3 : !py.dynamic
}

// CHECK-LABEL: func @__init__
// CHECK: %[[TYPE:.*]] = constant(#py.ref<@builtins.int>)
// CHECK: return %[[TYPE]]
