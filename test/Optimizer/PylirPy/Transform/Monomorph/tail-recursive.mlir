// RUN: pylir-opt %s --pylir-monomorph --split-input-file | FileCheck %s

// XFAIL: *

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

func @foo(%arg0: !py.dynamic) -> !py.dynamic {
	%0 = py.constant(#py.int<1>)
	%1 = py.int.cmp le %arg0, %0
	cf.cond_br %1, ^exit, ^recurse

^exit:
	return %0 : !py.dynamic

^recurse:
	%2 = py.constant(#py.int<-1>)
	%3 = py.int.add %2, %arg0
	%4 = py.call @foo(%3) : (!py.dynamic) -> !py.dynamic
	return %4 : !py.dynamic
}

func @__init__() {
	%0 = py.constant(#py.int<10>)
	%1 = py.call @foo(%0)
	test.use(%1) : !py.dynamic
	return
}

// CHECK-LABEL: func @__init__()
// CHECK: py.call @[[FOO_CLONE:.*]](%{{.*}}) : (!py.class<@builtins.int>) -> !py.unknown

// This should be optimized better in the future, but for now lets just make sure it terminates

// CHECK: func private @[[FOO_CLONE]]
