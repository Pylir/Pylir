// RUN: pylir-opt %s --test-type-flow --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type<slots = {__slots__ = #py.tuple<(#py.str<"__add__">)>}>
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.str = #py.type
py.globalValue const @builtins.dict = #py.type
py.globalValue const @builtins.function = #py.type
py.globalValue const @builtins.None = #py.type

func.func @builtins.int.__add__$impl(%closure : !py.dynamic, %tuple : !py.dynamic, %dict : !py.dynamic) -> !py.dynamic {
	%zero = arith.constant 0 : index
	%one = arith.constant 1 : index
	%first = py.tuple.getItem %tuple[%zero]
	%second = py.tuple.getItem %tuple[%one]
	%result = py.int.add %first, %second
	return %result : !py.dynamic
}

// CHECK-LABEL: typeFlow.func @builtins.int.__add__$impl
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG2:[[:alnum:]]+]]
// CHECK: %[[C0:.*]] = constant 0
// CHECK: %[[C1:.*]] = constant 1
// CHECK: %[[FIRST:.*]] = calc %[[ARG1]], %[[C0]]
// CHECK: %[[SECOND:.*]] = calc %[[ARG1]], %[[C1]]
// CHECK: %[[RES:.*]] = calc %[[FIRST]], %[[SECOND]]
// CHECK: return %[[RES]]

py.globalValue @builtins.int.__add__ = #py.function<@builtins.int.__add__$impl>
py.globalValue const @builtins.int = #py.type<slots = {__add__ = #py.ref<@builtins.int.__add__>}, mroTuple = #py.tuple<(#py.ref<@builtins.int>)>>

func.func @__init__() {
	%one = py.constant(#py.int<1>)
	%zero = py.constant(#py.int<0>)
	cf.br ^loop(%zero : !py.dynamic)

^loop(%iter : !py.dynamic):
	%0 = py.typeOf %iter
	%1 = py.type.mro %0
	%2 = py.mroLookup "__add__" in %1
	%3 = py.makeTuple (%iter, %one)
	%4 = py.constant(#py.dict<{}>)
	%5 = py.function.call %2(%2, %3, %4)
	%6 = test.random
	cf.cond_br %6, ^loop(%5 : !py.dynamic), ^exit

^exit:
	test.use(%5) : !py.dynamic
	py.unreachable
}

// CHECK-LABEL: typeFlow.func @__init__
// CHECK: %[[C0:.*]] = constant #py.int<1>
// CHECK: %[[C1:.*]] = constant #py.int<0>
// CHECK: branch ^[[BODY:[[:alnum:]]+]], (%[[C1]])
// CHECK: ^[[BODY]]
// CHECK-SAME: %[[ITER:[[:alnum:]]+]]
// CHECK-NEXT: %[[TYPE:.*]] = typeOf %[[ITER]]
// CHECK-NEXT: %[[MRO:.*]] = calc value %[[TYPE]]
// CHECK-SAME: py.type.mro
// CHECK-NEXT: %[[RESULT:.*]] = calc value %[[MRO]]
// CHECK-SAME: py.mroLookup "__add__"
// CHECK-NEXT: %[[TUPLE:.*]] = calc %[[ITER]], %[[C0]]
// CHECK-SAME: py.makeTuple
// CHECK-NEXT: %[[C2:.*]] = constant #py.dict<{}>
// CHECK-NEXT: %[[RES:.*]] = call_indirect %[[RESULT]](%[[RESULT]], %[[TUPLE]], %[[C2]])
// CHECK-NEXT: branch ^[[BODY]], ^[[EXIT:[[:alnum:]]+]], (%[[RES]])
// CHECK-NEXT: ^[[EXIT]]:
// CHECK-NEXT: branch

// -----

func.func @create1(%0 : !py.dynamic) -> !py.dynamic {
	%1 = py.makeObject %0
	return %1 : !py.dynamic
}

// CHECK-LABEL: typeFlow.func @create1
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK: %[[OBJ:.*]] = calc %[[ARG]]
// CHECK-SAME: py.makeObject
// CHECK-NEXT: return %[[OBJ]]

py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.tuple = #py.type

func.func @create2(%0 : !py.dynamic) -> !py.dynamic {
	%2 = py.constant(#py.tuple<()>)
	%1 = py.tuple.copy %2 : %0
	return %1 : !py.dynamic
}

// CHECK-LABEL: typeFlow.func @create2
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK: %[[C:.*]] = constant #py.tuple<()>
// CHECK: %[[OBJ:.*]] = calc %[[C]], %[[ARG]]
// CHECK-SAME: py.tuple.copy
// CHECK-NEXT: return %[[OBJ]]
