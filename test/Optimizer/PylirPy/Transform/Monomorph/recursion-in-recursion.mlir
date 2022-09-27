// RUN: pylir-opt %s --pylir-monomorph --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.int = #py.type

py.globalValue "private" @test

func.func @foo(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = test.random
	cf.cond_br %0, ^bb1, ^bb0

^bb0:
    %1 = py.call @foo(%arg0) : (!py.dynamic) -> !py.dynamic
    %2 = test.random
    cf.cond_br %2, ^bb1, ^bb2

^bb1:
    %3 = py.call @foo(%arg0) : (!py.dynamic) -> !py.dynamic
    return %3 : !py.dynamic

^bb2:
    return %arg0 : !py.dynamic
}

func.func @__init__() -> !py.dynamic {
	%1 = py.constant(#py.int<0>)
	%2 = py.call @foo(%1) : (!py.dynamic) -> !py.dynamic
	%3 = py.typeOf %2
	return %3 : !py.dynamic
}

// CHECK-LABEL: func @__init__
// CHECK: %[[RES:.*]] = py.call @foo
// CHECK: %[[TYPE:.*]] = py.typeOf %[[RES]]
// CHECK: return %[[TYPE]]
