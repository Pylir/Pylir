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

// CHECK-NOT: @foo

// CHECK-LABEL: @test
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
