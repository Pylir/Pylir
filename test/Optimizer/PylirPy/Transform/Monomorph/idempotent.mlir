// RUN: pylir-opt %s --pylir-monomorph --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type<slots = {__slots__ = #py.tuple<(#py.str<"__add__">)>}>
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.str = #py.type
py.globalValue const @builtins.dict = #py.type
py.globalValue const @builtins.function = #py.type
py.globalValue const @builtins.None = #py.type

func @builtins.int.__add__$impl(%arg0: !py.unknown, %arg1: !py.unknown, %arg2: !py.unknown) -> !py.unknown {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = py.tuple.getItem %arg1[%c0] : (!py.unknown) -> !py.unknown
  %1 = py.tuple.getItem %arg1[%c1] : (!py.unknown) -> !py.unknown
  %2 = py.int.add %0, %1 : !py.unknown, !py.unknown
  py.return %2 : !py.class<@builtins.int>
}

py.globalValue @builtins.int.__add__ = #py.function<@builtins.int.__add__$impl>
py.globalValue const @builtins.int = #py.type<mroTuple = #py.tuple<(@builtins.int)>, slots = {__add__ = @builtins.int.__add__}>

func @__init__() {
  %0 = py.constant(#py.int<1>) : !py.class<@builtins.int>
  %1 = py.constant(#py.int<0>) : !py.class<@builtins.int>
  py.br ^bb1(%1 : !py.class<@builtins.int>)

^bb1(%2: !py.class<@builtins.int>):  // 2 preds: ^bb0, ^bb1
  %3 = py.typeOf %2 : (!py.class<@builtins.int>) -> !py.unknown
  %4 = py.type.mro %3 : (!py.unknown) -> !py.class<@builtins.tuple>
  %result, %success = py.mroLookup "__add__" in %4 : (!py.class<@builtins.tuple>) -> !py.unknown
  %5 = py.makeTuple (%2, %0) : (!py.class<@builtins.int>, !py.class<@builtins.int>) -> !py.tuple<(!py.class<@builtins.int>, !py.class<@builtins.int>)>
  %6 = py.constant(#py.dict<{}>) : !py.class<@builtins.dict>
  %7 = py.call @builtins.int.__add__$impl_0(%result, %5, %6) : (!py.unknown, !py.tuple<(!py.class<@builtins.int>, !py.class<@builtins.int>)>, !py.class<@builtins.dict>) -> !py.class<@builtins.int>
  %8 = test.random
  py.cond_br %8, ^bb1(%7 : !py.class<@builtins.int>), ^bb2

^bb2:  // pred: ^bb1
  test.use(%7) : !py.class<@builtins.int>
  py.unreachable
}

func private @builtins.int.__add__$impl_0(%arg0: !py.unknown, %arg1: !py.tuple<(!py.class<@builtins.int>, !py.class<@builtins.int>)>, %arg2: !py.class<@builtins.dict>) -> !py.class<@builtins.int> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = py.tuple.getItem %arg1[%c0] : (!py.tuple<(!py.class<@builtins.int>, !py.class<@builtins.int>)>) -> !py.class<@builtins.int>
  %1 = py.tuple.getItem %arg1[%c1] : (!py.tuple<(!py.class<@builtins.int>, !py.class<@builtins.int>)>) -> !py.class<@builtins.int>
  %2 = py.int.add %0, %1 : !py.class<@builtins.int>, !py.class<@builtins.int>
  py.return %2 : !py.class<@builtins.int>
}

// CHECK-LABEL: func @builtins.int.__add__$impl
// CHECK-LABEL: func @__init__
// CHECK-LABEL: func private @builtins.int.__add__$impl_0
// CHECK-NOT: func {{.*}} @{{.*}}
