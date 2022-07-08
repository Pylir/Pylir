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

// CHECK: py.globalValue "private" @foo = #py.int<5>

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
