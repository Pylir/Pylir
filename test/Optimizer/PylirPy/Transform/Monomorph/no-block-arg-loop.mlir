// RUN: pylir-opt %s --pylir-monomorph --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue const @builtins.int = #py.type

func.func @__init__() -> !py.dynamic {
	%zero = py.constant(#py.int<0>)
	%one = py.constant(#py.int<1>)
	cf.br ^loop

^loop:
	%0 = test.random
	%1 = py.int.add %zero, %one
	cf.cond_br %0, ^loop, ^cont

^cont:
    %2 = test.random
    cf.cond_br %2, ^loop, ^exit

^exit:
    %3 = py.typeOf %1
	return %3 : !py.dynamic
}

// CHECK-LABEL: func.func @__init__
// CHECK: %[[TYPE:.*]] = py.constant(#py.ref<@builtins.int>)
// CHECK: return %[[TYPE]]

