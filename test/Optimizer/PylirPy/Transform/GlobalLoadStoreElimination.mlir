// RUN: pylir-opt %s -pass-pipeline='builtin.module(any(pylir-global-load-store-elimination))' --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type

py.global @foo : !py.dynamic

py.func @test() -> !py.dynamic {
    %0 = constant(#py.str<"value">)
    store %0 : !py.dynamic into @foo
    %1 = load @foo : !py.dynamic
    return %1 : !py.dynamic
}

// CHECK-LABEL:  func @test
// CHECK-NEXT: %[[C:.*]] = constant
// CHECK-NEXT: store %[[C]] : !py.dynamic into @foo
// CHECK-NEXT: return %[[C]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type

py.global @foo : !py.dynamic

py.func @test() -> !py.dynamic {
    %0 = constant(#py.str<"value">)
    %1 = constant(#py.str<"value">)
    store %0 : !py.dynamic into @foo
    store %1 : !py.dynamic into @foo
    %2 = load @foo : !py.dynamic
    return %2 : !py.dynamic
}

// CHECK-LABEL:  func @test
// CHECK-NEXT: constant
// CHECK-NEXT: %[[C:.*]] = constant
// CHECK-NEXT: store %[[C]] : !py.dynamic into @foo
// CHECK-NEXT: return %[[C]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type

py.func private @clobber()

py.global @foo : !py.dynamic

py.func @test() -> !py.dynamic {
    %0 = constant(#py.str<"value">)
    store %0 : !py.dynamic into @foo
    call @clobber() : () -> ()
    %1 = load @foo : !py.dynamic
    return %1 : !py.dynamic
}

// CHECK-LABEL:  func @test
// CHECK-NEXT: %[[C:.*]] = constant
// CHECK-NEXT: store %[[C]] : !py.dynamic into @foo
// CHECK-NEXT: call @clobber()
// CHECK-NEXT: %[[LOAD:.*]] = load @foo : !py.dynamic
// CHECK-NEXT: return %[[LOAD]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type

py.global @foo : !py.dynamic

py.func @test() -> !py.dynamic {
    %0 = test.random
    cf.cond_br %0, ^bb0, ^bb1

^bb0:
    %1 = constant(#py.str<"text">)
    store %1 : !py.dynamic into @foo
    cf.br ^merge

^bb1:
    %2 = constant(#py.str<"value">)
    store %2 : !py.dynamic into @foo
    cf.br ^merge

^merge:
    %3 = load @foo : !py.dynamic
    return %3 : !py.dynamic
}

// CHECK-LABEL: func @test
// CHECK: cf.cond_br %{{.*}}, ^[[FIRST:.*]], ^[[SECOND:[[:alnum:]]+]]

// CHECK: ^[[FIRST]]
// CHECK-NEXT: %[[C1:.*]] = constant(#py.str<"text">)
// CHECK-NEXT: store %[[C1]] : !py.dynamic into @foo
// CHECK-NEXT: cf.br ^[[MERGE:[[:alnum:]]+]]
// CHECK-SAME: %[[C1]]

// CHECK: ^[[SECOND]]:
// CHECK-NEXT: %[[C2:.*]] = constant(#py.str<"value">)
// CHECK-NEXT: store %[[C2]] : !py.dynamic into @foo
// CHECK-NEXT: cf.br ^[[MERGE]]
// CHECK-SAME: %[[C2]]

// CHECK: ^[[MERGE]]
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: return %[[ARG]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type

py.global @foo : !py.dynamic

py.func private @clobber()

py.func @test() -> !py.dynamic {
    %0 = test.random
    cf.cond_br %0, ^bb0, ^bb1

^bb0:
    call @clobber() : () -> ()
    cf.br ^merge

^bb1:
    %1 = constant(#py.str<"value">)
    store %1 : !py.dynamic into @foo
    cf.br ^merge

^merge:
    %2 = load @foo : !py.dynamic
    return %2 : !py.dynamic
}

// CHECK-LABEL: func @test
// CHECK: call @clobber
// CHECK-NEXT: cf.br ^[[MERGE:[[:alnum:]]+]]

// CHECK: store %{{.*}} : !py.dynamic into @foo
// CHECK-NEXT: cf.br ^[[MERGE]]

// CHECK: ^[[MERGE]]:
// CHECK-NEXT: %[[RESULT:.*]] = load @foo : !py.dynamic
// CHECK-NEXT: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type

py.global @foo : !py.dynamic

py.func @test() -> !py.dynamic {
    %0 = constant(#py.str<"value">)
    store %0 : !py.dynamic into @foo
    %r = test.random
    cf.cond_br %r, ^bb0, ^bb1

^bb0:
    %1 = constant(#py.str<"other">)
    store %1 : !py.dynamic into @foo
    cf.br ^bb1

^bb1:
    cf.br ^bb2

^bb2:
    %2 = load @foo : !py.dynamic
    return %2 : !py.dynamic
}

// CHECK-LABEL:  func @test
// CHECK: %[[C1:.*]] = constant(#py.str<"value">)
// CHECK: cf.cond_br %{{.*}}, ^{{.*}}, ^[[BB1:.*]](%[[C1]] : !py.dynamic)
// CHECK: %[[C2:.*]] = constant(#py.str<"other">)
// CHECK: cf.br ^[[BB1]](%[[C2]] : !py.dynamic)
// CHECK: ^[[BB1]](%[[ARG:.*]]: !py.dynamic):
// CHECK: return %[[ARG]]

// -----

module {
  py.globalValue @builtins.type = #py.type
  py.globalValue @builtins.int = #py.type
  py.globalValue @builtins.str = #py.type
  py.globalValue @builtins.tuple = #py.type
  py.globalValue @builtins.dict = #py.type
  py.globalValue @builtins.None = #py.type
  py.globalValue @builtins.next = #py.type
  py.globalValue @builtins.print = #py.type
  py.globalValue @builtins.bool = #py.type
  py.globalValue @builtins.BaseException.__new__ = #py.type
  py.globalValue @builtins.NameError = #py.type
  py.globalValue @builtins.StopIteration = #py.type
  py.globalValue @builtins.iter = #py.type
  py.func private @pylir__call__(!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic
  py.func private @builtins.__init__()
  py.func private @"builtins.BaseException.__new__$cc[0]"(!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic
  py.global "private" @it$handle : !py.dynamic
  py.global "private" @x$handle : !py.dynamic
  py.func @__init__() {
    %0 = constant(#py.unbound)
    %1 = constant(#py.ref<@builtins.iter>)
    %2 = constant(#py.dict<{}>)
    %3 = constant(#py.bool<True>)
    %4 = constant(#py.ref<@builtins.bool>)
    %5 = constant(#py.ref<@builtins.print>)
    %6 = constant(#py.ref<@builtins.next>)
    %7 = constant(#py.ref<@builtins.None>)
    %8 = constant(#py.ref<@builtins.StopIteration>)
    %true = arith.constant true
    %9 = constant(#py.tuple<(#py.tuple<(#py.int<3>, #py.int<5>, #py.int<6>)>)>)
    %10 = constant(#py.tuple<(#py.ref<@builtins.NameError>)>)
    %11 = constant(#py.ref<@builtins.BaseException.__new__>)
    %12 = constant(#py.tuple<(#py.tuple<(#py.int<2>, #py.int<4>, #py.int<7>)>)>)
    call @builtins.__init__() : () -> ()
    store %0 : !py.dynamic into @x$handle
    store %0 : !py.dynamic into @it$handle
    %13 = call @pylir__call__(%1, %9, %2) : (!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic
    store %13 : !py.dynamic into @it$handle
    cf.br ^bb1(%3 : !py.dynamic)
  ^bb1(%14: !py.dynamic):  // 2 preds: ^bb0, ^bb5
    %15 = py.bool.toI1 %14
    cf.cond_br %15, ^bb2, ^bb8
  ^bb2:  // pred: ^bb1
    %16 = load @it$handle : !py.dynamic
    %17 = isUnboundValue %16
    cf.cond_br %17, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %18 = call @"builtins.BaseException.__new__$cc[0]"(%11, %10, %2) : (!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic
    %19 = typeOf %18
    cf.br ^bb6(%18 : !py.dynamic)
  ^bb4:  // pred: ^bb2
    %20 = makeTuple (%16)
    %21 = invoke @pylir__call__(%6, %20, %2) : (!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic
      label ^bb5 unwind ^bb6
  ^bb5:  // pred: ^bb4
    %22 = makeTuple (%21)
    %23 = invoke @pylir__call__(%5, %22, %2) : (!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic
      label ^bb1(%3 : !py.dynamic) unwind ^bb6
  ^bb6(%24: !py.dynamic):  // 3 preds: ^bb3, ^bb4, ^bb5
    %25 = typeOf %24
    %26 = py.type.mro %25
    %27 = py.tuple.contains %8 in %26
    cf.cond_br %27, ^bb8, ^bb7
  ^bb7:  // pred: ^bb6
    raise %24
  ^bb8:  // 2 preds: ^bb1, ^bb6
    %28 = call @pylir__call__(%1, %12, %2) : (!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic
    store %28 : !py.dynamic into @it$handle
    cf.br ^bb9
  ^bb9:  // 2 preds: ^bb8, ^bb20
    %29 = load @it$handle : !py.dynamic
    %30 = isUnboundValue %29
    cf.cond_br %30, ^bb10, ^bb11
  ^bb10:  // 3 preds: ^bb9, ^bb14, ^bb19
    %31 = call @"builtins.BaseException.__new__$cc[0]"(%11, %10, %2) : (!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic
    %32 = typeOf %31
    raise %31
  ^bb11:  // pred: ^bb9
    %33 = makeTuple (%29, %7)
    %34 = call @pylir__call__(%6, %33, %2) : (!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic
    store %34 : !py.dynamic into @x$handle
    %35 = typeOf %34
    %36 = is %35, %4
    cf.cond_br %36, ^bb13(%34 : !py.dynamic), ^bb12
  ^bb12:  // pred: ^bb11
    %37 = makeTuple (%34)
    %38 = call @pylir__call__(%4, %37, %2) : (!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic
    cf.br ^bb13(%38 : !py.dynamic)
  ^bb13(%39: !py.dynamic):  // 2 preds: ^bb11, ^bb12
    %40 = py.bool.toI1 %39
    cf.cond_br %40, ^bb14, ^bb16(%34 : !py.dynamic)
  ^bb14:  // pred: ^bb13
    %41 = load @x$handle : !py.dynamic
    %42 = isUnboundValue %41
    cf.cond_br %42, ^bb10, ^bb15
  ^bb15:  // pred: ^bb14
    %43 = is %41, %7
    %44 = arith.xori %43, %true : i1
    %45 = py.bool.fromI1 %44
    cf.br ^bb16(%45 : !py.dynamic)
  ^bb16(%46: !py.dynamic):  // 2 preds: ^bb13, ^bb15
    %47 = typeOf %46
    %48 = is %47, %4
    cf.cond_br %48, ^bb18(%46 : !py.dynamic), ^bb17
  ^bb17:  // pred: ^bb16
    %49 = makeTuple (%46)
    %50 = call @pylir__call__(%4, %49, %2) : (!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic
    cf.br ^bb18(%50 : !py.dynamic)
  ^bb18(%51: !py.dynamic):  // 2 preds: ^bb16, ^bb17
    %52 = py.bool.toI1 %51
    cf.cond_br %52, ^bb19, ^bb21
  ^bb19:  // pred: ^bb18
    %53 = load @x$handle : !py.dynamic
    %54 = isUnboundValue %53
    cf.cond_br %54, ^bb10, ^bb20
  ^bb20:  // pred: ^bb19
    %55 = makeTuple (%53)
    %56 = call @pylir__call__(%5, %55, %2) : (!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic
    cf.br ^bb9
  ^bb21:  // pred: ^bb18
    return
  }
}

// CHECK-LABEL:  py.func @__init__

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type

py.global @foo : !py.dynamic

py.func @test() -> !py.dynamic {
    %0 = constant(#py.str<"value">)
    store %0 : !py.dynamic into @foo
    cf.br ^bb0

^bb0:
    %1 = load @foo : !py.dynamic
    test.use(%1) : !py.dynamic // acts as a clobber because it has unknown side effects.
    %2 = test.random
    cf.cond_br %2, ^bb0, ^bb2

^bb2:
    %3 = load @foo : !py.dynamic
    return %3 : !py.dynamic
}

// CHECK-LABEL:  func @test
// CHECK: %[[C1:.*]] = constant(#py.str<"value">)
// CHECK: store %[[C1]] : !py.dynamic into @foo
// CHECK: cf.br ^[[BB0:[[:alnum:]]+]]

// CHECK: ^[[BB0]]:
// CHECK: load @foo : !py.dynamic
// CHECK: cf.cond_br %{{.*}}, ^[[BB0]], ^[[BB2:[[:alnum:]]+]]

// CHECK: ^[[BB2]]:
// CHECK: %[[LOAD:.*]] = load @foo : !py.dynamic
// CHECK: return %[[LOAD]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type

py.global @foo : !py.dynamic

py.func @test() -> !py.dynamic {
    %0 = constant(#py.str<"value">)
    store %0 : !py.dynamic into @foo
    cf.br ^bb0

^bb0:
    %1 = load @foo : !py.dynamic
    %2 = test.random
    cf.cond_br %2, ^bb0, ^bb2

^bb2:
    %3 = load @foo : !py.dynamic
    return %3 : !py.dynamic
}

// CHECK-LABEL:  func @test
// CHECK: %[[C:.*]] = constant(#py.str<"value">)
// CHECK: cf.br ^[[BB0:[[:alnum:]]+]](%[[C]] : !py.dynamic)

// CHECK: ^[[BB0]](%[[ARG:.*]]: !py.dynamic):
// CHECK-NOT: load
// CHECK: cf.cond_br %{{.*}}, ^[[BB0]](%[[ARG]] : !py.dynamic), ^[[BB2:[[:alnum:]]+]]

// CHECK: ^[[BB2]]:
// CHECK-NOT: load
// CHECK: return %[[ARG]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type

py.global @foo : !py.dynamic

py.func @test(%arg0 : !py.dynamic) -> !py.dynamic {
    store %arg0 : !py.dynamic into @foo
    %c = arith.constant 5 : index
    py.list.resize %arg0 to %c
    %1 = load @foo : !py.dynamic
    return %1 : !py.dynamic
}

// CHECK-LABEL:  func @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: store %[[ARG0]] : !py.dynamic into @foo
// CHECK-NOT: load
// CHECK: return %[[ARG0]]
