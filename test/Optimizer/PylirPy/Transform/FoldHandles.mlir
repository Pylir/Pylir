// RUN: pylir-opt %s --fold-handles --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

py.globalHandle "private" @foo

func @test() {
    %0 = py.constant(#py.int<5>) : !py.unknown
    py.store %0 into @foo : !py.unknown
    return
}

func @bar() -> !py.unknown {
    %0 = py.load @foo : !py.unknown
    return %0 : !py.unknown
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

func @test() {
    %0 = py.constant(#py.int<5>) : !py.unknown
    py.store %0 into @foo : !py.unknown
    return
}

// CHECK-NOT: @foo

// CHECK-LABEL: @test
// CHECK-NOT: py.store %{{.*}} into @foo
// CHECK: return
