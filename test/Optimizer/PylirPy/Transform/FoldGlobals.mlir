// RUN: pylir-opt %s --pylir-fold-globals --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

py.global "private" @foo : !py.dynamic

func.func @test() {
    %0 = py.constant(#py.int<5>)
    py.store %0 : !py.dynamic into @foo
    return
}

func.func @bar() -> !py.dynamic {
    %0 = py.load @foo : !py.dynamic
    return %0 : !py.dynamic
}

// CHECK: py.globalValue "private" const @foo = #py.int<5>

// CHECK-LABEL: @test
// CHECK-NOT: py.store %{{.*}} into @foo
// CHECK: return

// CHECK-LABEL: @bar
// CHECK-NEXT: %[[C:.*]] = py.constant(#py.ref<@foo>)
// CHECK-NEXT: return %[[C]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

py.global "private" @foo : !py.dynamic

func.func @test() {
    %0 = py.constant(#py.int<5>)
    py.store %0 : !py.dynamic into @foo
    return
}

func.func @bar() {
    %0 = py.constant(#py.int<10>)
    py.store %0 : !py.dynamic into @foo
    return
}

// CHECK-NOT: @foo

// CHECK-LABEL: @test
// CHECK-NOT: py.store %{{.*}} into @foo
// CHECK: return

// CHECK-LABEL: @bar
// CHECK-NOT: py.store %{{.*}} into @foo
// CHECK: return

// -----


py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

py.global "private" @foo : !py.dynamic

func.func @test() {
    %0 = py.constant(#py.ref<@builtins.int>)
    py.store %0 : !py.dynamic into @foo
    return
}

func.func @bar() -> !py.dynamic {
    %0 = py.load @foo : !py.dynamic
    return %0 : !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-NOT: py.store %{{.*}} : !py.dynamic into @foo
// CHECK: return

// CHECK-LABEL: @bar
// CHECK-NEXT: %[[C:.*]] = py.constant(#py.ref<@builtins.int>)
// CHECK-NEXT: return %[[C]]

// -----


py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

py.global "private" @foo : !py.dynamic

func.func @test() {
    %0 = py.constant(#py.unbound)
    py.store %0 : !py.dynamic into @foo
    return
}

func.func @bar() -> !py.dynamic {
    %0 = py.load @foo : !py.dynamic
    return %0 : !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-NOT: py.store %{{.*}} : !py.dynamic into @foo
// CHECK: return

// CHECK-LABEL: @bar
// CHECK-NEXT: %[[C:.*]] = py.constant(#py.unbound)
// CHECK-NEXT: return %[[C]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.function = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.dict = #py.type
py.globalValue @builtins.None = #py.type
py.globalValue @builtins.str = #py.type

func.func @real(%arg0 : !py.dynamic, %arg1 : !py.dynamic, %arg2 : !py.dynamic) -> !py.dynamic {
    %0 = py.typeOf %arg0
    return %0 : !py.dynamic
}

py.global "private" @foo : !py.dynamic

func.func @test() {
    %0 = py.makeFunc @real
    py.store %0 : !py.dynamic into @foo
    return
}

func.func @bar() -> !py.dynamic {
    %0 = py.load @foo : !py.dynamic
    return %0 : !py.dynamic
}

// CHECK: py.globalValue "private" @foo = #py.function<@real>

// CHECK-LABEL: @test
// CHECK-NOT: py.makeFunc
// CHECK-NOT: py.store %{{.*}} into @foo
// CHECK: return

// CHECK-LABEL: @bar
// CHECK-NEXT: %[[C:.*]] = py.constant(#py.ref<@foo>)
// CHECK-NEXT: return %[[C]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

py.global "private" @foo : !py.dynamic

func.func @test() -> !py.dynamic {
    %0 = py.constant(#py.int<5>)
    py.store %0 : !py.dynamic into @foo
    %1 = test.random
    cf.cond_br %1, ^bb1, ^bb2

^bb1:
    %2 = py.constant(#py.int<10>)
    py.store %2 : !py.dynamic into @foo
    cf.br ^bb3

^bb2:
    %3 = py.load @foo : !py.dynamic
    %4 = py.int.add %3, %0
    py.store %4 : !py.dynamic into @foo
    cf.br ^bb3

^bb3:
    %5 = py.load @foo : !py.dynamic
    return %5 : !py.dynamic
}

// CHECK-NOT: @foo

// CHECK-LABEL: func.func @test
// CHECK-NEXT: %[[C0:.*]] = py.constant(#py.int<5>)
// CHECK-NEXT: %[[R:.*]] = test.random
// CHECK-NEXT: cf.cond_br %[[R]], ^[[BB1:.*]], ^[[BB2:[[:alnum:]]+]]

// CHECK: ^[[BB1]]:
// CHECK-NEXT: %[[C2:.*]] = py.constant(#py.int<10>)
// CHECK-NEXT: cf.br ^[[BB3:.*]](%[[C2]] : !py.dynamic)

// CHECK: ^[[BB2]]:
// CHECK-NEXT: %[[RES:.*]] = py.int.add %[[C0]], %[[C0]]
// CHECK-NEXT: cf.br ^[[BB3:.*]](%[[RES]] : !py.dynamic)

// CHECK: ^[[BB3]](%[[ARG:.*]]: !py.dynamic):
// CHECK-NEXT: return %[[ARG]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

py.global "private" @bar : !py.dynamic
py.global "private" @foo : !py.dynamic

func.func @test() {
    %0 = py.constant(#py.int<5>)
    py.store %0 : !py.dynamic into @foo
    return
}

func.func @other() {
    %0 = py.load @foo : !py.dynamic
    py.store %0 : !py.dynamic into @bar
    return
}

func.func @other2() -> !py.dynamic {
    %0 = py.load @bar : !py.dynamic
    return %0 : !py.dynamic
}

// CHECK-NOT: @bar
// CHECK-LABEL: func.func @test
// CHECK-NOT: py.store
// CHECK: return

// CHECK-LABEL: func.func @other
// CHECK-NOT: py.load
// CHECK-NOT: py.store
// CHECK: return

// CHECK-LABEL: func.func @other2
// CHECK-NOT: py.load
// CHECK: return

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

py.global "private" @foo : index

func.func @test() {
    %0 = arith.constant 5 : index
    py.store %0 : index into @foo
    return
}

func.func @bar() -> index {
    %0 = py.load @foo : index
    return %0 : index
}

// CHECK-LABEL: @test
// CHECK-NOT: py.store
// CHECK: return

// CHECK-LABEL: @bar
// CHECK-NEXT: %[[C:.*]] = arith.constant 5
// CHECK-NEXT: return %[[C]]

// -----

py.global "private" @foo : index

func.func @bar() -> index {
    %0 = py.load @foo : index
    return %0 : index
}

// CHECK-LABEL: @bar
// CHECK-NEXT: %[[C:.*]] = arith.constant
// CHECK-NEXT: return %[[C]]


// -----

py.global "private" @foo : index = 5 : index

func.func @bar() -> index {
    %0 = py.load @foo : index
    return %0 : index
}

// CHECK-LABEL: @bar
// CHECK-NEXT: %[[C:.*]] = arith.constant 5 : index
// CHECK-NEXT: return %[[C]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

py.global "private" @foo : index = 3 : index

func.func @test() {
    %0 = arith.constant 5 : index
    py.store %0 : index into @foo
    return
}

func.func @bar() -> index {
    %0 = py.load @foo : index
    return %0 : index
}

// CHECK-LABEL: @test
// CHECK: py.store

// CHECK-LABEL: @bar
// CHECK: py.load


// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

py.global "private" @foo : index = 5 : index

func.func @test() {
    %0 = arith.constant 5 : index
    py.store %0 : index into @foo
    return
}

func.func @bar() -> index {
    %0 = py.load @foo : index
    return %0 : index
}

// CHECK-LABEL: @test
// CHECK-NOT: py.store
// CHECK: return

// CHECK-LABEL: @bar
// CHECK-NEXT: %[[C:.*]] = arith.constant 5
// CHECK-NEXT: return %[[C]]
