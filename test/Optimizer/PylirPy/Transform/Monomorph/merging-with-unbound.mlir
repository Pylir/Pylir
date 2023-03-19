// RUN: pylir-opt %s --pylir-monomorph --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.int = #py.type

py.func @__init__() -> !py.dynamic {
	%1 = constant(#py.unbound)
	%2 = constant(#py.int<0>)
	%3 = test.random
    cf.cond_br %3, ^body, ^skip(%1 : !py.dynamic)

^body:
    test.use(%2) : !py.dynamic
    cf.br ^skip(%2 : !py.dynamic)

^skip(%4 : !py.dynamic):
	%5 = typeOf %4
	return %5 : !py.dynamic
}

// CHECK-LABEL: func @__init__
// CHECK: %[[TYPE:.*]] = constant(#py.ref<@builtins.int>)
// CHECK: return %[[TYPE]]
