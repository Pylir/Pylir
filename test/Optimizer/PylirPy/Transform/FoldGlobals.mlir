// RUN: pylir-opt %s --pylir-fold-globals --split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int

py.global "private" @foo : !py.dynamic

py.func @test() {
    %0 = constant(#py.int<5>)
    store %0 : !py.dynamic into @foo
    return
}

py.func @bar() -> !py.dynamic {
    %0 = load @foo : !py.dynamic
    return %0 : !py.dynamic
}

// CHECK: #[[$FOO:.*]] = #py.globalValue<foo, const, initializer = #py.int<5>>

// CHECK-LABEL: @test
// CHECK-NOT: store %{{.*}} into @foo
// CHECK: return

// CHECK-LABEL: @bar
// CHECK-NEXT: %[[C:.*]] = constant(#[[$FOO]])
// CHECK-NEXT: return %[[C]]

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int

py.global "private" @foo : !py.dynamic

py.func @test() {
    %0 = constant(#py.int<5>)
    store %0 : !py.dynamic into @foo
    return
}

py.func @bar() {
    %0 = constant(#py.int<10>)
    store %0 : !py.dynamic into @foo
    return
}

// CHECK-NOT: @foo

// CHECK-LABEL: @test
// CHECK-NOT: store %{{.*}} into @foo
// CHECK: return

// CHECK-LABEL: @bar
// CHECK-NOT: store %{{.*}} into @foo
// CHECK: return

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int

py.global "private" @foo : !py.dynamic

py.func @test() {
    %0 = constant(#builtins_int)
    store %0 : !py.dynamic into @foo
    return
}

py.func @bar() -> !py.dynamic {
    %0 = load @foo : !py.dynamic
    return %0 : !py.dynamic
}

// CHECK: #[[$INT:.*]] = #py.globalValue<builtins.int{{,|>}}

// CHECK-LABEL: @test
// CHECK-NOT: store %{{.*}} : !py.dynamic into @foo
// CHECK: return

// CHECK-LABEL: @bar
// CHECK-NEXT: %[[C:.*]] = constant(#[[$INT]])
// CHECK-NEXT: return %[[C]]

// -----


#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int

py.global "private" @foo : !py.dynamic

py.func @test() {
    %0 = constant(#py.unbound)
    store %0 : !py.dynamic into @foo
    return
}

py.func @bar() -> !py.dynamic {
    %0 = load @foo : !py.dynamic
    return %0 : !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-NOT: store %{{.*}} : !py.dynamic into @foo
// CHECK: return

// CHECK-LABEL: @bar
// CHECK-NEXT: %[[C:.*]] = constant(#py.unbound)
// CHECK-NEXT: return %[[C]]

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int
#builtins_function = #py.globalValue<builtins.function, initializer = #py.type>
py.external @builtins.function, #builtins_function
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_dict= #py.globalValue<builtins.dict, initializer = #py.type>
py.external @builtins.dict, #builtins_dict
#builtins_None = #py.globalValue<builtins.None, initializer = #py.type>
py.external @builtins.None, #builtins_None
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str

py.func @real(%arg0 : !py.dynamic, %arg1 : !py.dynamic, %arg2 : !py.dynamic) -> !py.dynamic {
    %0 = typeOf %arg0
    return %0 : !py.dynamic
}

py.global "private" @foo : !py.dynamic

py.func @test() {
    %0 = makeFunc @real
    store %0 : !py.dynamic into @foo
    return
}

py.func @bar() -> !py.dynamic {
    %0 = load @foo : !py.dynamic
    return %0 : !py.dynamic
}

// CHECK: #[[$FOO:.*]] = #py.globalValue<foo, initializer = #py.function<@real>>

// CHECK-LABEL: @test
// CHECK-NOT: makeFunc
// CHECK-NOT: store %{{.*}} into @foo
// CHECK: return

// CHECK-LABEL: @bar
// CHECK-NEXT: %[[C:.*]] = constant(#[[$FOO]])
// CHECK-NEXT: return %[[C]]

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int

py.global "private" @foo : !py.dynamic

py.func @test() -> !py.dynamic {
    %0 = constant(#py.int<5>)
    store %0 : !py.dynamic into @foo
    %1 = test.random
    cf.cond_br %1, ^bb1, ^bb2

^bb1:
    %2 = constant(#py.int<10>)
    store %2 : !py.dynamic into @foo
    cf.br ^bb3

^bb2:
    %3 = load @foo : !py.dynamic
    %4 = int_add %3, %0
    store %4 : !py.dynamic into @foo
    cf.br ^bb3

^bb3:
    %5 = load @foo : !py.dynamic
    return %5 : !py.dynamic
}

// CHECK-NOT: @foo

// CHECK-LABEL: py.func @test
// CHECK-NEXT: %[[C0:.*]] = constant(#py.int<5>)
// CHECK-NEXT: %[[R:.*]] = test.random
// CHECK-NEXT: cf.cond_br %[[R]], ^[[BB1:.*]], ^[[BB2:[[:alnum:]]+]]

// CHECK: ^[[BB1]]:
// CHECK-NEXT: %[[C2:.*]] = constant(#py.int<10>)
// CHECK-NEXT: cf.br ^[[BB3:.*]](%[[C2]] : !py.dynamic)

// CHECK: ^[[BB2]]:
// CHECK-NEXT: %[[RES:.*]] = int_add %[[C0]], %[[C0]]
// CHECK-NEXT: cf.br ^[[BB3:.*]](%[[RES]] : !py.dynamic)

// CHECK: ^[[BB3]](%[[ARG:.*]]: !py.dynamic):
// CHECK-NEXT: return %[[ARG]]

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int

py.global "private" @bar : !py.dynamic
py.global "private" @foo : !py.dynamic

py.func @test() {
    %0 = constant(#py.int<5>)
    store %0 : !py.dynamic into @foo
    return
}

py.func @other() {
    %0 = load @foo : !py.dynamic
    store %0 : !py.dynamic into @bar
    return
}

py.func @other2() -> !py.dynamic {
    %0 = load @bar : !py.dynamic
    return %0 : !py.dynamic
}

// CHECK-NOT: @bar
// CHECK-LABEL: py.func @test
// CHECK-NOT: store
// CHECK: return

// CHECK-LABEL: py.func @other
// CHECK-NOT: load
// CHECK-NOT: store
// CHECK: return

// CHECK-LABEL: py.func @other2
// CHECK-NOT: load
// CHECK: return

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int

py.global "private" @foo : index

py.func @test() {
    %0 = arith.constant 5 : index
    store %0 : index into @foo
    return
}

py.func @bar() -> index {
    %0 = load @foo : index
    return %0 : index
}

// CHECK-LABEL: @test
// CHECK-NOT: store
// CHECK: return

// CHECK-LABEL: @bar
// CHECK-NEXT: %[[C:.*]] = arith.constant 5
// CHECK-NEXT: return %[[C]]

// -----

py.global "private" @foo : index

py.func @bar() -> index {
    %0 = load @foo : index
    return %0 : index
}

// CHECK-LABEL: @bar
// CHECK-NEXT: %[[C:.*]] = arith.constant
// CHECK-NEXT: return %[[C]]


// -----

py.global "private" @foo : index = 5 : index

py.func @bar() -> index {
    %0 = load @foo : index
    return %0 : index
}

// CHECK-LABEL: @bar
// CHECK-NEXT: %[[C:.*]] = arith.constant 5 : index
// CHECK-NEXT: return %[[C]]

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int

py.global "private" @foo : index = 3 : index

py.func @test() {
    %0 = arith.constant 5 : index
    store %0 : index into @foo
    return
}

py.func @bar() -> index {
    %0 = load @foo : index
    return %0 : index
}

// CHECK-LABEL: @test
// CHECK: store

// CHECK-LABEL: @bar
// CHECK: load


// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int

py.global "private" @foo : index = 5 : index

py.func @test() {
    %0 = arith.constant 5 : index
    store %0 : index into @foo
    return
}

py.func @bar() -> index {
    %0 = load @foo : index
    return %0 : index
}

// CHECK-LABEL: @test
// CHECK-NOT: store
// CHECK: return

// CHECK-LABEL: @bar
// CHECK-NEXT: %[[C:.*]] = arith.constant 5
// CHECK-NEXT: return %[[C]]
