// RUN: pylir-opt %s --pylir-monomorph --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.int = #py.type

// TODO: If we have some kind of observable support for variant types this test should make use of it

py.func @foo(%arg0 : !py.dynamic) -> !py.dynamic {
	%0 = test.random
	cf.cond_br %0, ^call, ^exit(%arg0 : !py.dynamic)

^call:
	%1 = constant(#py.int<1>)
	%2 = int_add %arg0, %1
	%3 = call @foo(%2) : (!py.dynamic) -> !py.dynamic
	cf.br ^exit(%3 : !py.dynamic)

^exit(%4 : !py.dynamic):
	%5 = constant(#py.int<2>)
    %6 = int_add %4, %5
	return %6 : !py.dynamic
}

py.func @__init__() -> !py.dynamic {
	%1 = constant(#py.int<0>)
	%2 = call @foo(%1) : (!py.dynamic) -> !py.dynamic
	%3 = typeOf %2
	return %3 : !py.dynamic
}

// CHECK-LABEL: func @__init__
// CHECK: %[[RES:.*]] = call @foo
// CHECK: %[[TYPE:.*]] = typeOf %[[RES]]
// CHECK: return %[[TYPE]]
