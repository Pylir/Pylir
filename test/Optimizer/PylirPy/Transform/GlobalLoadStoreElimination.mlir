// RUN: pylir-opt %s -pass-pipeline='builtin.module(any(pylir-global-load-store-elimination))' --split-input-file | FileCheck %s

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

py.global @foo : !py.dynamic

py.func @test(%arg0 : !py.dynamic) -> !py.dynamic {
    store %arg0 : !py.dynamic into @foo
    %c = arith.constant 5 : index
    list_resize %arg0 to %c
    %1 = load @foo : !py.dynamic
    return %1 : !py.dynamic
}

// CHECK-LABEL:  func @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: store %[[ARG0]] : !py.dynamic into @foo
// CHECK-NOT: load
// CHECK: return %[[ARG0]]
