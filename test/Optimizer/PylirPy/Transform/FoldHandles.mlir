// RUN: pylir-opt %s --pylir-fold-handles --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

py.globalHandle "private" @foo

func.func @test() {
    %0 = py.constant(#py.int<5>)
    py.store %0 into @foo
    return
}

func.func @bar() -> !py.dynamic {
    %0 = py.load @foo
    return %0 : !py.dynamic
}

// CHECK: py.globalValue "private" const @foo = #py.int<5>

// CHECK-LABEL: @test
// CHECK-NOT: py.store %{{.*}} into @foo
// CHECK: return

// CHECK-LABEL: @bar
// CHECK-NEXT: %[[C:.*]] = py.constant(@foo)
// CHECK-NEXT: return %[[C]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

py.globalHandle "private" @foo

func.func @test() {
    %0 = py.constant(#py.int<5>)
    py.store %0 into @foo
    return
}

func.func @bar() {
    %0 = py.constant(#py.int<10>)
    py.store %0 into @foo
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

py.globalHandle "private" @foo

func.func @test() {
    %0 = py.constant(@builtins.int)
    py.store %0 into @foo
    return
}

func.func @bar() -> !py.dynamic {
    %0 = py.load @foo
    return %0 : !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-NOT: py.store %{{.*}} into @foo
// CHECK: return

// CHECK-LABEL: @bar
// CHECK-NEXT: %[[C:.*]] = py.constant(@builtins.int)
// CHECK-NEXT: return %[[C]]

// -----


py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

py.globalHandle "private" @foo

func.func @test() {
    %0 = py.constant(#py.unbound)
    py.store %0 into @foo
    return
}

func.func @bar() -> !py.dynamic {
    %0 = py.load @foo
    return %0 : !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-NOT: py.store %{{.*}} into @foo
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

py.globalHandle "private" @foo

func.func @test() {
    %0 = py.makeFunc @real
    py.store %0 into @foo
    return
}

func.func @bar() -> !py.dynamic {
    %0 = py.load @foo
    return %0 : !py.dynamic
}

// CHECK: py.globalValue "private" @foo = #py.function<@real>

// CHECK-LABEL: @test
// CHECK-NOT: py.makeFunc
// CHECK-NOT: py.store %{{.*}} into @foo
// CHECK: return

// CHECK-LABEL: @bar
// CHECK-NEXT: %[[C:.*]] = py.constant(@foo)
// CHECK-NEXT: return %[[C]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

py.globalHandle "private" @foo

func.func @test() -> !py.dynamic {
    %0 = py.constant(#py.int<5>)
    py.store %0 into @foo
    %1 = test.random
    cf.cond_br %1, ^bb1, ^bb2

^bb1:
    %2 = py.constant(#py.int<10>)
    py.store %2 into @foo
    cf.br ^bb3

^bb2:
    %3 = py.load @foo
    %4 = py.int.add %3, %0
    py.store %4 into @foo
    cf.br ^bb3

^bb3:
    %5 = py.load @foo
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

py.globalHandle "private" @bar
py.globalHandle "private" @foo

func.func @test() {
    %0 = py.constant(#py.int<5>)
    py.store %0 into @foo
    return
}

func.func @other() {
    %0 = py.load @foo
    py.store %0 into @bar
    return
}

func.func @other2() -> !py.dynamic {
    %0 = py.load @bar
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
