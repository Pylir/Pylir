// RUN: pylir-opt %s --pylir-monomorph --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.int = #py.type

func.func @fibonacci(%arg0 : !py.dynamic) -> !py.dynamic {
	%0 = py.constant(#py.int<1>)
	%1 = py.int.cmp le %arg0, %0
	cf.cond_br %1, ^returnOne, ^recurse

^returnOne:
	return %0 : !py.dynamic

^recurse:
	%2 = py.constant(#py.int<-2>)
	%3 = py.constant(#py.int<-1>)
    %4 = py.int.add %arg0, %3
    %5 = py.int.add %arg0, %2
    %6 = py.call @fibonacci(%4) : (!py.dynamic) -> !py.dynamic
    %7 = py.call @fibonacci(%5) : (!py.dynamic) -> !py.dynamic
    %8 = py.int.add %6, %7
	return %8 : !py.dynamic
}

func.func @__init__() -> !py.dynamic {
	%1 = py.constant(#py.int<10>)
	%2 = py.call @fibonacci(%1) : (!py.dynamic) -> !py.dynamic
	%3 = py.typeOf %2
	return %3 : !py.dynamic
}

// CHECK-LABEL: func @__init__
// CHECK: %[[RES:.*]] = py.call @fibonacci
// CHECK: %[[TYPE:.*]] = py.typeOf %[[RES]]
// CHECK: return %[[TYPE]]
