// RUN: pylir-opt %s --pylir-monomorph --canonicalize --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue const @builtins.int = #py.type

func.func @__init__() -> !py.dynamic {
	%zero = py.constant(#py.int<0>)
	%one = py.constant(#py.int<1>)
	cf.br ^loop(%zero : !py.dynamic)

^loop(%iter : !py.dynamic):
	%0 = test.random
	%1 = py.int.add %iter, %one
	cf.cond_br %0, ^loop(%1 : !py.dynamic), ^cont

^cont:
    %2 = test.random
    cf.cond_br %2, ^loop(%zero : !py.dynamic), ^exit

^exit:
    %3 = py.typeOf %iter
	return %3 : !py.dynamic
}

// CHECK-LABEL: func.func @__init__
// CHECK: %[[TYPE:.*]] = py.constant(#py.ref<@builtins.int>)
// CHECK: return %[[TYPE]]

