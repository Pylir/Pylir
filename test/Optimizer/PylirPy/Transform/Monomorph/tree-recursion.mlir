// RUN: pylir-opt %s --pylir-monomorph --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.dict = #py.type
py.globalValue @builtins.BaseException = #py.type
py.globalValue @builtins.TypeError = #py.type
py.globalValue @builtins.bool = #py.type
py.globalValue @builtins.None = #py.type
py.globalValue @builtins.function = #py.type
py.globalValue @builtins.str = #py.type

py.func private @"builtins.type.__call__$impl[0]"(%arg0: !py.dynamic, %arg1: !py.dynamic, %arg2: !py.dynamic, %arg3: !py.dynamic) -> !py.dynamic {
  %0 = constant(#py.dict<{}>)
  %1 = constant(#py.ref<@builtins.None>)
  %2 = constant(#py.int<1>)
  %3 = constant(#py.int<0>)
  %4 = constant(#py.ref<@builtins.bool>)
  %true = arith.constant true
  %5 = constant(#py.ref<@builtins.type>)
  %6 = constant(#py.ref<@builtins.BaseException>)
  %c0 = arith.constant 0 : index
  %7 = constant(#py.tuple<()>)
  %8 = constant(#py.ref<@builtins.TypeError>)
  %9 = constant(#py.ref<@builtins.type.__call__>)
  %10 = py.tuple.len %arg2
  %11 = py.int.fromUnsigned %10
  %12 = py.dict.len %arg3
  %13 = py.int.fromUnsigned %12
  %14 = py.int.cmp eq %11, %2
  %15 = py.bool.fromI1 %14
  %16 = py.int.cmp eq %13, %3
  %17 = py.bool.fromI1 %16
  %18 = is %arg1, %5
  %19 = py.bool.fromI1 %18
  %20 = arith.select %18, %15, %19 : !py.dynamic
  %21 = typeOf %20
  %22 = test.random
  cf.cond_br %22, ^bb3(%20 : !py.dynamic), ^bb2
^bb1:  // pred: ^bb10
  %23 = makeObject %8
  raise %23
^bb2:  // pred: ^bb0
  %24 = makeTuple (%20)
  %25 = call @"builtins.type.__call__$impl[0]"(%9, %4, %24, %0) : (!py.dynamic, !py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic
  cf.br ^bb3(%25 : !py.dynamic)
^bb3(%26: !py.dynamic):  // 2 preds: ^bb0, ^bb2
  %27 = py.bool.toI1 %26
  %28 = arith.select %27, %17, %20 : !py.dynamic
  %29 = typeOf %28
  %30 = test.random
  cf.cond_br %30, ^bb5(%28 : !py.dynamic), ^bb4
^bb4:  // pred: ^bb3
  %31 = makeTuple (%28)
  %32 = call @"builtins.type.__call__$impl[0]"(%9, %4, %31, %0) : (!py.dynamic, !py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic
  cf.br ^bb5(%32 : !py.dynamic)
^bb5(%33: !py.dynamic):  // 2 preds: ^bb3, ^bb4
  %34 = py.bool.toI1 %33
  cf.cond_br %34, ^bb6, ^bb7
^bb6:  // pred: ^bb5
  %35 = py.tuple.getItem %arg2[%c0]
  %36 = typeOf %35
  return %36 : !py.dynamic
^bb7:  // pred: ^bb5
  %37 = py.type.mro %arg1
  %result = mroLookup %c0 in %37
  %40 = py.tuple.prepend %arg1, %arg2
  %41 = py.function.call %result(%result, %40, %arg3)
  %42 = typeOf %41
  %43 = py.type.mro %42
  %44 = py.tuple.contains %arg1 in %43
  %45 = test.random
  cf.cond_br %45, ^bb8, ^bb9
^bb8:  // 2 preds: ^bb7, ^bb9
  return %41 : !py.dynamic
^bb9:  // pred: ^bb7
  %result_0 = mroLookup %c0 in %43
  %48 = py.tuple.prepend %41, %arg2
  %49 = py.function.call %result_0(%result_0, %48, %arg3)
  %50 = is %49, %1
  %51 = test.random
  cf.cond_br %51, ^bb10, ^bb8
^bb10:  // pred: ^bb9
  %52 = call @"builtins.type.__call__$impl[0]"(%9, %8, %7, %0) : (!py.dynamic, !py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic
  %53 = typeOf %52
  %54 = py.type.mro %53
  %55 = py.tuple.contains %6 in %54
  cf.cond_br %55, ^bb11, ^bb1
^bb11:  // pred: ^bb10
  raise %52
}

py.func private @"builtins.type.__call__$cc[0]"(%arg0: !py.dynamic, %arg1: !py.dynamic, %arg2: !py.dynamic) -> !py.dynamic {
  %c0 = arith.constant 0 : index
  %0 = constant(#py.unbound)
  %1 = constant(#py.str<"self">)
  %2 = constant(#py.dict<{}>)
  %3 = constant(#py.ref<@builtins.None>)
  %c1 = arith.constant 1 : index
  %5 = constant(#py.ref<@builtins.TypeError>)
  %6 = constant(#py.tuple<()>)
  %7 = py.tuple.len %arg1
  %8 = arith.cmpi ugt, %7, %c0 : index
  cf.cond_br %8, ^bb1, ^bb2(%0 : !py.dynamic)
^bb1:  // pred: ^bb0
  %9 = py.tuple.getItem %arg1[%c0]
  cf.br ^bb2(%9 : !py.dynamic)
^bb2(%10: !py.dynamic):  // 2 preds: ^bb0, ^bb1
  %11 = py.dict.tryGetItem %arg2[%1 hash(%c0)]
  %12 = isUnboundValue %11
  cf.cond_br %12, ^bb5(%10 : !py.dynamic), ^bb3
^bb3:  // pred: ^bb2
  %13 = py.dict.delItem %1 hash(%c0) from %arg2
  %14 = isUnboundValue %10
  cf.cond_br %14, ^bb5(%11 : !py.dynamic), ^bb4
^bb4:  // 2 preds: ^bb3, ^bb5
  %15 = makeObject %5
  %16 = typeOf %15
  raise %15
^bb5(%17: !py.dynamic):  // 2 preds: ^bb2, ^bb3
  %18 = isUnboundValue %17
  cf.cond_br %18, ^bb4, ^bb6
^bb6:  // pred: ^bb5
  %19 = py.tuple.dropFront %c1, %arg1
  %20 = call @"builtins.type.__call__$impl[0]"(%arg0, %17, %19, %arg2) : (!py.dynamic, !py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic
  return %20 : !py.dynamic
}

py.globalValue "private" const @builtins.type.__call__ = #py.function<@"builtins.type.__call__$cc[0]", qual_name = #py.str<"builtins.type.__call__">, defaults = #py.tuple<()>, kw_defaults = #py.dict<{}>>

py.func @root() -> !py.dynamic {
    %3 = constant(#py.dict<{}>)
    %7 = constant(#py.tuple<()>)
    %10 = constant(#py.ref<@builtins.type.__call__>)
    %12 = constant(#py.ref<@builtins.TypeError>)
    %17 = call @"builtins.type.__call__$impl[0]"(%10, %12, %7, %3) : (!py.dynamic, !py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic
    return %17 : !py.dynamic
}

// CHECK-LABEL: py.func @root

