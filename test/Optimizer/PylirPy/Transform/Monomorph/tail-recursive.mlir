// RUN: pylir-opt %s --pylir-monomorph --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.int = #py.type

py.func @foo(%arg0: !py.dynamic) -> !py.dynamic {
	%0 = constant(#py.int<1>)
	%1 = int_cmp le %arg0, %0
	cf.cond_br %1, ^exit, ^recurse

^exit:
	return %0 : !py.dynamic

^recurse:
	%2 = constant(#py.int<-1>)
	%3 = int_add %2, %arg0
	%4 = call @foo(%3) : (!py.dynamic) -> !py.dynamic
	return %4 : !py.dynamic
}

py.func @__init__() -> !py.dynamic {
	%0 = constant(#py.int<10>)
	%1 = call @foo(%0) : (!py.dynamic) -> !py.dynamic
	%2 = typeOf %1
	return %2 : !py.dynamic
}

// CHECK-LABEL: func @__init__()
// CHECK: %[[RES:.*]] = call @foo
// CHECK: %[[TYPE:.*]] = typeOf %[[RES]]
// CHECK: return %[[TYPE]]
