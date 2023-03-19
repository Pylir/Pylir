// RUN: pylir-opt %s --pylir-monomorph --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.int = #py.type

py.globalValue "private" @test

py.func @foo(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = test.random
	cf.cond_br %0, ^bb1, ^bb0

^bb0:
    %1 = call @foo(%arg0) : (!py.dynamic) -> !py.dynamic
    %2 = test.random
    cf.cond_br %2, ^bb1, ^bb2

^bb1:
    %3 = call @foo(%arg0) : (!py.dynamic) -> !py.dynamic
    return %3 : !py.dynamic

^bb2:
    return %arg0 : !py.dynamic
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
