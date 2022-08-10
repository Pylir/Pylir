// RUN: pylir-opt %s --pylir-monomorph --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

func.func @foo(%arg0: !py.dynamic) -> !py.dynamic {
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

func.func @__init__() -> !py.dynamic {
	%0 = py.constant(#py.int<10>)
	%1 = py.call @foo(%0) : (!py.dynamic) -> !py.dynamic
	%2 = py.typeOf %1
	return %2 : !py.dynamic
}

// CHECK-LABEL: func @__init__()
// CHECK: %[[RES:.*]] = py.call @foo
// CHECK: %[[TYPE:.*]] = py.typeOf %[[RES]]
// CHECK: return %[[TYPE]]
