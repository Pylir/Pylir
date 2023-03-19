// RUN: pylir-opt %s --pylir-monomorph --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue const @builtins.int = #py.type

py.func @__init__() -> !py.dynamic {
	%zero = constant(#py.int<0>)
	%one = constant(#py.int<1>)
	cf.br ^loop

^loop:
	%0 = test.random
	%1 = py.int.add %zero, %one
	cf.cond_br %0, ^loop, ^cont

^cont:
    %2 = test.random
    cf.cond_br %2, ^loop, ^exit

^exit:
    %3 = typeOf %1
	return %3 : !py.dynamic
}

// CHECK-LABEL: py.func @__init__
// CHECK: %[[TYPE:.*]] = constant(#py.ref<@builtins.int>)
// CHECK: return %[[TYPE]]

