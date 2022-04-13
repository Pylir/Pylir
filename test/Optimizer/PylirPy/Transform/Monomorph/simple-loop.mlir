// RUN: pylir-opt %s --pylir-monomorph --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type<slots = {__slots__ = #py.tuple<(#py.str<"__add__">)>}>
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.str = #py.type
py.globalValue const @builtins.dict = #py.type
py.globalValue const @builtins.function = #py.type
py.globalValue const @builtins.None = #py.type

func @builtins.int.__add__$impl(%closure : !py.unknown, %tuple : !py.unknown, %dict : !py.unknown) -> !py.unknown {
	%zero = arith.constant 0 : index
	%one = arith.constant 1 : index
	%first = py.tuple.getItem %tuple[%zero] : (!py.unknown) -> !py.unknown
	%second = py.tuple.getItem %tuple[%one] : (!py.unknown) -> !py.unknown
	%result = py.int.add %first, %second : !py.unknown, !py.unknown
	py.return %result : !py.class<@builtins.int>
}

py.globalValue @builtins.int.__add__ = #py.function<@builtins.int.__add__$impl>
py.globalValue const @builtins.int = #py.type<slots = {__add__ = @builtins.int.__add__}, mroTuple = #py.tuple<(@builtins.int)>>

func @__init__() {
	%one = py.constant(#py.int<1>) : !py.unknown
	%zero = py.constant(#py.int<0>) : !py.unknown
	py.br ^loop(%zero : !py.unknown)

^loop(%iter : !py.unknown):
	%0 = py.typeOf %iter : (!py.unknown) -> !py.unknown
	%1 = py.type.mro %0 : (!py.unknown) -> !py.unknown
	%2, %found = py.mroLookup "__add__" in %1 : (!py.unknown) -> !py.unknown
	%3 = py.makeTuple (%iter, %one) : (!py.unknown, !py.unknown) -> !py.unknown
	%4 = py.constant(#py.dict<{}>) : !py.unknown
	%5 = py.function.call %2(%2, %3, %4) : !py.unknown(!py.unknown, !py.unknown, !py.unknown) -> !py.unknown
	%6 = test.random
	py.cond_br %6, ^loop(%5 : !py.unknown), ^exit

^exit:
	test.use(%5) : !py.unknown
	py.unreachable
}

// CHECK-LABEL: func @__init__
// CHECK-DAG: %[[ONE:.*]] = py.constant(#py.int<1>) : !py.class<@builtins.int>
// CHECK-DAG: %[[ZERO:.*]] = py.constant(#py.int<0>) : !py.class<@builtins.int>
// CHECK-DAG: %[[INT_TYPE:.*]] = py.constant(@builtins.int)
// CHECK: py.br ^[[LOOP:.*]](%[[ZERO]] : !py.class<@builtins.int>)
// CHECK-NEXT: ^[[LOOP]]
// CHECK-SAME: %[[ITER:[[:alnum:]]+]]: !py.class<@builtins.int>
// CHECK-NEXT: %[[MRO:.*]] = py.type.mro %[[INT_TYPE]]
// CHECK-NEXT: %[[LOOKUP:.*]], %{{.*}} = py.mroLookup "__add__" in %[[MRO]]
// CHECK-NEXT: %[[TUPLE:.*]] = py.makeTuple (%[[ITER]], %[[ONE]])
// CHECK-NEXT: %[[DICT:.*]] = py.constant(#py.dict<{}>)
// CHECK-NEXT: %[[RESULT:.*]] = py.call @[[CLONED:.*]](%[[LOOKUP]], %[[TUPLE]], %[[DICT]])
// CHECK: py.cond_br %{{.*}}, ^[[LOOP]](%[[RESULT]] : !py.class<@builtins.int>), ^[[EXIT:[[:alnum:]]+]]
// CHECK-NEXT: ^[[EXIT]]:
// CHECK-NEXT: test.use(%[[RESULT]]) : !py.class<@builtins.int>

// CHECK: func private @[[CLONED]]
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[TUPLE:[[:alnum:]]+]]: !py.tuple<(!py.class<@builtins.int>, !py.class<@builtins.int>)>
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: -> !py.class<@builtins.int>
// CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0
// CHECK-NEXT: %[[ONE:.*]] = arith.constant 1
// CHECK-NEXT: %[[FIRST:.*]] = py.tuple.getItem %[[TUPLE]][%[[ZERO]]]
// CHECK-SAME: -> !py.class<@builtins.int>
// CHECK-NEXT: %[[SECOND:.*]] = py.tuple.getItem %[[TUPLE]][%[[ONE]]]
// CHECK-SAME: -> !py.class<@builtins.int>
// CHECK-NEXT: %[[RESULT:.*]] = py.int.add %[[FIRST]], %[[SECOND]]
// CHECK-NEXT: py.return %[[RESULT]] : !py.class<@builtins.int>
