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

func.func private @"builtins.type.__call__$impl[0]"(%arg0: !py.dynamic, %arg1: !py.dynamic, %arg2: !py.dynamic, %arg3: !py.dynamic) -> !py.dynamic {
  %0 = py.constant(#py.dict<{}>)
  %1 = py.constant(@builtins.None)
  %2 = py.constant(#py.int<1>)
  %3 = py.constant(#py.int<0>)
  %4 = py.constant(@builtins.bool)
  %true = arith.constant true
  %5 = py.constant(@builtins.type)
  %6 = py.constant(@builtins.BaseException)
  %c0 = arith.constant 0 : index
  %7 = py.constant(#py.tuple<()>)
  %8 = py.constant(@builtins.TypeError)
  %9 = py.constant(@builtins.type.__call__)
  %10 = py.tuple.len %arg2
  %11 = py.int.fromUnsigned %10
  %12 = py.dict.len %arg3
  %13 = py.int.fromUnsigned %12
  %14 = py.int.cmp eq %11, %2
  %15 = py.bool.fromI1 %14
  %16 = py.int.cmp eq %13, %3
  %17 = py.bool.fromI1 %16
  %18 = py.is %arg1, %5
  %19 = py.bool.fromI1 %18
  %20 = arith.select %18, %15, %19 : !py.dynamic
  %21 = py.typeOf %20
  %22 = test.random
  cf.cond_br %22, ^bb3(%20 : !py.dynamic), ^bb2
^bb1:  // pred: ^bb10
  %23 = py.makeObject %8
  py.raise %23
^bb2:  // pred: ^bb0
  %24 = py.makeTuple (%20)
  %25 = py.call @"builtins.type.__call__$impl[0]"(%9, %4, %24, %0) : (!py.dynamic, !py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic
  cf.br ^bb3(%25 : !py.dynamic)
^bb3(%26: !py.dynamic):  // 2 preds: ^bb0, ^bb2
  %27 = py.bool.toI1 %26
  %28 = arith.select %27, %17, %20 : !py.dynamic
  %29 = py.typeOf %28
  %30 = test.random
  cf.cond_br %30, ^bb5(%28 : !py.dynamic), ^bb4
^bb4:  // pred: ^bb3
  %31 = py.makeTuple (%28)
  %32 = py.call @"builtins.type.__call__$impl[0]"(%9, %4, %31, %0) : (!py.dynamic, !py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic
  cf.br ^bb5(%32 : !py.dynamic)
^bb5(%33: !py.dynamic):  // 2 preds: ^bb3, ^bb4
  %34 = py.bool.toI1 %33
  cf.cond_br %34, ^bb6, ^bb7
^bb6:  // pred: ^bb5
  %35 = py.tuple.getItem %arg2[%c0]
  %36 = py.typeOf %35
  return %36 : !py.dynamic
^bb7:  // pred: ^bb5
  %37 = py.type.mro %arg1
  %result = py.mroLookup "__new__" in %37
  %40 = py.tuple.prepend %arg1, %arg2
  %41 = py.function.call %result(%result, %40, %arg3)
  %42 = py.typeOf %41
  %43 = py.type.mro %42
  %44 = py.tuple.contains %arg1 in %43
  %45 = test.random
  cf.cond_br %45, ^bb8, ^bb9
^bb8:  // 2 preds: ^bb7, ^bb9
  return %41 : !py.dynamic
^bb9:  // pred: ^bb7
  %result_0 = py.mroLookup "__init__" in %43
  %48 = py.tuple.prepend %41, %arg2
  %49 = py.function.call %result_0(%result_0, %48, %arg3)
  %50 = py.is %49, %1
  %51 = test.random
  cf.cond_br %51, ^bb10, ^bb8
^bb10:  // pred: ^bb9
  %52 = py.call @"builtins.type.__call__$impl[0]"(%9, %8, %7, %0) : (!py.dynamic, !py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic
  %53 = py.typeOf %52
  %54 = py.type.mro %53
  %55 = py.tuple.contains %6 in %54
  cf.cond_br %55, ^bb11, ^bb1
^bb11:  // pred: ^bb10
  py.raise %52
}

func.func private @"builtins.type.__call__$cc[0]"(%arg0: !py.dynamic, %arg1: !py.dynamic, %arg2: !py.dynamic) -> !py.dynamic {
  %c0 = arith.constant 0 : index
  %0 = py.constant(#py.unbound)
  %1 = py.constant(#py.str<"self">)
  %2 = py.constant(#py.dict<{}>)
  %3 = py.constant(@builtins.None)
  %c1 = arith.constant 1 : index
  %5 = py.constant(@builtins.TypeError)
  %6 = py.constant(#py.tuple<()>)
  %7 = py.tuple.len %arg1
  %8 = arith.cmpi ugt, %7, %c0 : index
  cf.cond_br %8, ^bb1, ^bb2(%0 : !py.dynamic)
^bb1:  // pred: ^bb0
  %9 = py.tuple.getItem %arg1[%c0]
  cf.br ^bb2(%9 : !py.dynamic)
^bb2(%10: !py.dynamic):  // 2 preds: ^bb0, ^bb1
  %11 = py.dict.tryGetItem %arg2[%1 hash(%c0)]
  %12 = py.isUnboundValue %11
  cf.cond_br %12, ^bb5(%10 : !py.dynamic), ^bb3
^bb3:  // pred: ^bb2
  %13 = py.dict.delItem %1 hash(%c0) from %arg2
  %14 = py.isUnboundValue %10
  cf.cond_br %14, ^bb5(%11 : !py.dynamic), ^bb4
^bb4:  // 2 preds: ^bb3, ^bb5
  %15 = py.makeObject %5
  %16 = py.typeOf %15
  py.setSlot "__context__" of %15 : %16 to %3
  py.setSlot "__cause__" of %15 : %16 to %3
  py.raise %15
^bb5(%17: !py.dynamic):  // 2 preds: ^bb2, ^bb3
  %18 = py.isUnboundValue %17
  cf.cond_br %18, ^bb4, ^bb6
^bb6:  // pred: ^bb5
  %19 = py.tuple.dropFront %c1, %arg1
  %20 = py.call @"builtins.type.__call__$impl[0]"(%arg0, %17, %19, %arg2) : (!py.dynamic, !py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic
  return %20 : !py.dynamic
}

py.globalValue "private" const @builtins.type.__call__ = #py.function<@"builtins.type.__call__$cc[0]", qualName = #py.str<"builtins.type.__call__">, defaults = #py.tuple<()>, kwDefaults = #py.dict<{}>>

func.func @root() -> !py.dynamic {
    %3 = py.constant(#py.dict<{}>)
    %7 = py.constant(#py.tuple<()>)
    %10 = py.constant(@builtins.type.__call__)
    %12 = py.constant(@builtins.TypeError)
    %17 = py.call @"builtins.type.__call__$impl[0]"(%10, %12, %7, %3) : (!py.dynamic, !py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic
    return %17 : !py.dynamic
}

// CHECK-LABEL: func.func @root

