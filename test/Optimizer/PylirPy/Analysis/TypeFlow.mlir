// RUN: pylir-opt %s --test-type-flow --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type<slots = {__slots__ = #py.tuple<(#py.str<"__add__">)>}>
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.str = #py.type
py.globalValue const @builtins.dict = #py.type
py.globalValue const @builtins.function = #py.type
py.globalValue const @builtins.None = #py.type

func @builtins.int.__add__$impl(%closure : !py.dynamic, %tuple : !py.dynamic, %dict : !py.dynamic) -> !py.dynamic {
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
py.globalValue const @builtins.int = #py.type<slots = {__add__ = @builtins.int.__add__}, mroTuple = #py.tuple<(@builtins.int)>>

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

// CHECK-LABEL: typeFlow.func @__init__
// CHECK-DAG: %[[C0:.*]] = constant #py.int<1>
// CHECK-DAG: %[[C1:.*]] = constant #py.int<0>
// CHECK: loop(%[[ITER:.*]] = %[[C1]]) -> {
// CHECK-NEXT: %[[TYPE:.*]] = typeOf %[[ITER]]
// CHECK-NEXT: %[[MRO:.*]] = calc value %[[TYPE]]
// CHECK-SAME: py.type.mro
// CHECK-NEXT: %[[RESULT:.*]]:2 = calc value %[[MRO]]
// CHECK-SAME: py.mroLookup "__add__"
// CHECK-NEXT: %[[TUPLE:.*]] = calc %[[ITER]], %[[C0]]
// CHECK-SAME: py.makeTuple
// CHECK-NEXT: %[[C2:.*]] = constant #py.dict<{}>
// CHECK-NEXT: %[[RES:.*]] = call_indirect %[[RESULT]]#0(%[[RESULT]]#0, %[[TUPLE]], %[[C2]])
// CHECK-NEXT: branch ^[[YIELD:.*]], ^[[EXIT:[[:alnum:]]+]]
// CHECK-NEXT: ^[[YIELD]]:
// CHECK-NEXT: yield(%[[RES]])
// CHECK-NEXT: ^[[EXIT]]:
// CHECK-NEXT: exit 0
// CHECK-NEXT: } successors ^[[EXIT:[[:alnum:]]+]]
// CHECK-NEXT: ^[[EXIT]]:
// CHECK-NEXT: branch

// -----

func @create1(%0 : !py.dynamic) -> !py.dynamic {
	%1 = py.makeObject %0
	return %1 : !py.dynamic
}

// CHECK-LABEL: typeFlow.func @create1
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK: %[[OBJ:.*]] = makeObject %[[ARG]]
// CHECK-NEXT: return %[[OBJ]]

py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.tuple = #py.type

func @create2(%0 : !py.dynamic) -> !py.dynamic {
	%2 = py.constant(#py.tuple<()>)
	%1 = py.tuple.copy %2 : %0
	return %1 : !py.dynamic
}

// CHECK-LABEL: typeFlow.func @create2
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK: %[[OBJ:.*]] = makeObject %[[ARG]]
// CHECK-NEXT: return %[[OBJ]]

// -----


py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.dict = #py.type
py.globalValue const @builtins.str = #py.type
py.globalValue const @builtins.None = #py.type
py.globalValue const @builtins.str.__new__ = #py.type

func private @builtins.str.__new__$impl(!py.dynamic, !py.dynamic, !py.dynamic ,!py.dynamic) -> !py.dynamic

func private @builtins.print$impl(%arg0: !py.dynamic, %arg1: !py.dynamic, %arg2: !py.dynamic, %arg3: !py.dynamic) -> !py.dynamic {
  %0 = py.constant(#py.dict<{}>)
  %1 = py.constant(@builtins.str)
  %c0 = arith.constant 0 : index
  %2 = py.constant(#py.str<"">)
  %c1 = arith.constant 1 : index
  %3 = py.constant(#py.str<"\0A">)
  %4 = py.constant(#py.str<" ">)
  %5 = py.constant(@builtins.None)
  %6 = py.constant(@builtins.type)
  %7 = py.constant(@builtins.str.__new__)
  %8 = py.is %arg2, %5
  %9 = arith.select %8, %4, %arg2 : !py.dynamic
  %10 = py.is %arg3, %5
  %11 = arith.select %10, %3, %arg3 : !py.dynamic
  %12 = py.tuple.len %arg1
  %13 = arith.cmpi ult, %12, %c1 : index
  cf.cond_br %13, ^bb9(%2 : !py.dynamic), ^bb1
^bb1:  // pred: ^bb0
  %14 = py.tuple.getItem %arg1[%c0]
  %15 = py.is %1, %6
  cf.cond_br %15, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  %16 = py.typeOf %14
  cf.br ^bb4(%16, %c1 : !py.dynamic, index)
^bb3:  // pred: ^bb1
  %17 = py.makeTuple (%14)
  %18 = py.call @builtins.str.__new__$impl(%7, %1, %17, %0) : (!py.dynamic, !py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic
  cf.br ^bb4(%18, %c1 : !py.dynamic, index)
^bb4(%19: !py.dynamic, %20: index):  // 3 preds: ^bb2, ^bb3, ^bb8
  %21 = arith.cmpi ult, %20, %12 : index
  cf.cond_br %21, ^bb5, ^bb9(%19 : !py.dynamic)
^bb5:  // pred: ^bb4
  %22 = py.tuple.getItem %arg1[%20]
  %23 = py.is %1, %6
  cf.cond_br %23, ^bb6, ^bb7
^bb6:  // pred: ^bb5
  %24 = py.typeOf %22
  cf.br ^bb8(%24 : !py.dynamic)
^bb7:  // pred: ^bb5
  %25 = py.makeTuple (%22)
  %26 = py.call @builtins.str.__new__$impl(%7, %1, %25, %0) : (!py.dynamic, !py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic
  cf.br ^bb8(%26 : !py.dynamic)
^bb8(%27: !py.dynamic):  // 2 preds: ^bb6, ^bb7
  %28 = py.str.concat %19, %9, %27
  %29 = arith.addi %20, %c1 : index
  cf.br ^bb4(%28, %29 : !py.dynamic, index)
^bb9(%30: !py.dynamic):  // 2 preds: ^bb0, ^bb4
  %31 = py.str.concat %30, %11
  py.intr.print %31
  return %5 : !py.dynamic
}

// CHECK-LABEL: typeFlow.func @builtins.print$impl
// Just testing it doesn't crash at the moment
