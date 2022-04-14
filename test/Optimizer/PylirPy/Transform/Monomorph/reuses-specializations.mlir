// RUN: pylir-opt %s --pylir-monomorph --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type<slots = {__slots__ = #py.tuple<(#py.str<"__add__">)>}>
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.str = #py.type
py.globalValue const @builtins.dict = #py.type
py.globalValue const @builtins.function = #py.type
py.globalValue const @builtins.None = #py.type

func @builtins.int.__add__$impl(%arg0: !py.dynamic, %arg1: !py.dynamic, %arg2: !py.dynamic) -> !py.dynamic {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = py.tuple.getItem %arg1[%c0]
  %1 = py.tuple.getItem %arg1[%c1]
  %2 = py.int.add %0, %1
  return %2 : !py.dynamic
}

py.globalValue @builtins.int.__add__ = #py.function<@builtins.int.__add__$impl>
py.globalValue const @builtins.int = #py.type<mroTuple = #py.tuple<(@builtins.int)>, slots = {__add__ = @builtins.int.__add__}>

func @__init__() {
	%one = py.constant(#py.int<1>)
	%zero = py.constant(#py.int<0>)
	cf.br ^loop(%zero : !py.dynamic)

^loop(%iter : !py.dynamic):
	%0 = py.typeOf %iter
	%1 = py.type.mro %0
	%2, %found = py.mroLookup "__add__" in %1
	%3 = py.makeTuple (%iter, %one)
	%4 = py.constant(#py.dict<{}>)
	%5 = py.function.call %2(%2, %3, %4)
	%6 = test.random
	cf.cond_br %6, ^loop(%5 : !py.dynamic), ^exit

^exit:
	test.use(%5) : !py.dynamic
	py.unreachable
}

func private @builtins.int.__add__$impl_0(%arg0: !py.dynamic {py.specialization_args = !py.unknown}, %arg1: !py.dynamic {py.specialization_args = !py.tuple<(!py.class<@builtins.int>, !py.class<@builtins.int>)>}, %arg2: !py.dynamic {py.specialization_args = !py.class<@builtins.dict>}) -> (!py.dynamic {py.specialization_args = !py.class<@builtins.int>}) attributes {py.specialization_of = "builtins.int.__add__$impl"} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = py.tuple.getItem %arg1[%c0]
  %1 = py.tuple.getItem %arg1[%c1]
  %2 = py.int.add %0, %1
  return %2 : !py.dynamic
}

// CHECK-LABEL: func @__init__
// CHECK-DAG: %[[ONE:.*]] = py.constant(#py.int<1>)
// CHECK-DAG: %[[ZERO:.*]] = py.constant(#py.int<0>)
// CHECK-DAG: %[[INT_TYPE:.*]] = py.constant(@builtins.int)
// CHECK: cf.br ^[[LOOP:.*]](%[[ZERO]] : {{.*}})
// CHECK-NEXT: ^[[LOOP]]
// CHECK-SAME: %[[ITER:[[:alnum:]]+]]
// CHECK-NEXT: %[[MRO:.*]] = py.type.mro %[[INT_TYPE]]
// CHECK-NEXT: %[[LOOKUP:.*]], %{{.*}} = py.mroLookup "__add__" in %[[MRO]]
// CHECK-NEXT: %[[TUPLE:.*]] = py.makeTuple (%[[ITER]], %[[ONE]])
// CHECK-NEXT: %[[DICT:.*]] = py.constant(#py.dict<{}>)
// CHECK-NEXT: %[[RESULT:.*]] = py.call @builtins.int.__add__$impl_0(%[[LOOKUP]], %[[TUPLE]], %[[DICT]])
// CHECK: cf.cond_br %{{.*}}, ^[[LOOP]](%[[RESULT]] : {{.*}}), ^[[EXIT:[[:alnum:]]+]]
// CHECK-NEXT: ^[[EXIT]]:
// CHECK-NEXT: test.use(%[[RESULT]])

// CHECK-LABEL: func private @builtins.int.__add__$impl_0
// CHECK-NOT: func {{.*}} @{{.*}}
