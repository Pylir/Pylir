// RUN: pylir-opt %s --pylir-monomorph --canonicalize --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type<instance_slots = #py.tuple<(#py.str<"__add__">)>>
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.str = #py.type
py.globalValue const @builtins.dict = #py.type
py.globalValue const @builtins.function = #py.type
py.globalValue const @builtins.None = #py.type

py.func @builtins.int.__add__$impl(%closure : !py.dynamic, %tuple : !py.dynamic, %dict : !py.dynamic) -> !py.dynamic {
	%zero = arith.constant 0 : index
	%one = arith.constant 1 : index
	%first = tuple_getItem %tuple[%zero]
	%second = tuple_getItem %tuple[%one]
	%result = int_add %first, %second
	return %result : !py.dynamic
}

py.globalValue @builtins.int.__add__ = #py.function<@builtins.int.__add__$impl>
py.globalValue const @builtins.int = #py.type<slots = {__add__ = #py.ref<@builtins.int.__add__>}, mro_tuple = #py.tuple<(#py.ref<@builtins.int>)>>

py.func @__init__() {
	%one = constant(#py.int<1>)
	%zero = constant(#py.int<0>)
	%c0 = arith.constant 0 : index
	cf.br ^loop(%zero : !py.dynamic)

^loop(%iter : !py.dynamic):
	%0 = typeOf %iter
	%1 = type_mro %0
	%2 = mroLookup %c0 in %1
	%3 = makeTuple (%iter, %one)
	%4 = constant(#py.dict<{}>)
	%5 = function_call %2(%2, %3, %4)
	%6 = test.random
	cf.cond_br %6, ^loop(%5 : !py.dynamic), ^exit

^exit:
	test.use(%5) : !py.dynamic
	py.unreachable
}

// CHECK-LABEL: func @__init__
// CHECK-DAG: %[[ONE:.*]] = constant(#py.int<1>)
// CHECK-DAG: %[[ZERO:.*]] = constant(#py.int<0>)
// CHECK-DAG: %[[DICT:.*]] = constant(#py.dict<{}>)
// CHECK-DAG: %[[FUNC:.*]] = constant(#py.ref<@builtins.int.__add__>)
// CHECK: cf.br ^[[LOOP:.*]](%[[ZERO]] : {{.*}})
// CHECK-NEXT: ^[[LOOP]]
// CHECK-SAME: %[[ITER:[[:alnum:]]+]]
// CHECK-NEXT: %[[TUPLE:.*]] = makeTuple (%[[ITER]], %[[ONE]])
// CHECK-NEXT: %[[RESULT:.*]] = call @builtins.int.__add__$impl(%[[FUNC]], %[[TUPLE]], %[[DICT]])
// CHECK: cf.cond_br %{{.*}}, ^[[LOOP]](%[[RESULT]] : {{.*}}), ^[[EXIT:[[:alnum:]]+]]
// CHECK-NEXT: ^[[EXIT]]:
// CHECK-NEXT: test.use(%[[RESULT]])
