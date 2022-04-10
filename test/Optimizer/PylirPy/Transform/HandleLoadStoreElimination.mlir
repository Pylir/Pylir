// RUN: pylir-opt %s -pass-pipeline='func.func(pylir-handle-load-store-elimination)' --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type

py.globalHandle @foo

func @test() -> !py.unknown {
    %0 = py.constant(#py.str<"value">) : !py.unknown
    py.store %0 into @foo : !py.unknown
    %1 = py.load @foo : !py.unknown
    return %1 : !py.unknown
}

// CHECK-LABEL:  func @test
// CHECK-NEXT: %[[C:.*]] = py.constant
// CHECK-NEXT: py.store %[[C]] into @foo
// CHECK-NEXT: return %[[C]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type

py.globalHandle @foo

func @test() -> !py.unknown {
    %0 = py.constant(#py.str<"value">) : !py.unknown
    %1 = py.constant(#py.str<"value">) : !py.unknown
    py.store %0 into @foo : !py.unknown
    py.store %1 into @foo : !py.unknown
    %2 = py.load @foo : !py.unknown
    return %2 : !py.unknown
}

// CHECK-LABEL:  func @test
// CHECK-NEXT: py.constant
// CHECK-NEXT: %[[C:.*]] = py.constant
// CHECK-NEXT: py.store %[[C]] into @foo
// CHECK-NEXT: return %[[C]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type

func private @clobber()

py.globalHandle @foo

func @test() -> !py.unknown {
    %0 = py.constant(#py.str<"value">) : !py.unknown
    py.store %0 into @foo : !py.unknown
    call @clobber() : () -> ()
    %1 = py.load @foo : !py.unknown
    return %1 : !py.unknown
}

// CHECK-LABEL:  func @test
// CHECK-NEXT: %[[C:.*]] = py.constant
// CHECK-NEXT: py.store %[[C]] into @foo
// CHECK-NEXT: call @clobber()
// CHECK-NEXT: %[[LOAD:.*]] = py.load @foo
// CHECK-NEXT: return %[[LOAD]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type

py.globalHandle @foo

func @test() -> !py.unknown {
    %0 = test.random
    cf.cond_br %0, ^bb0, ^bb1

^bb0:
    %1 = py.constant(#py.str<"text">) : !py.unknown
    py.store %1 into @foo : !py.unknown
    cf.br ^merge

^bb1:
    %2 = py.constant(#py.str<"value">) : !py.unknown
    py.store %2 into @foo : !py.unknown
    cf.br ^merge

^merge:
    %3 = py.load @foo : !py.unknown
    return %3 : !py.unknown
}

// CHECK-LABEL: func @test
// CHECK: cf.cond_br %{{.*}}, ^[[FIRST:.*]], ^[[SECOND:[[:alnum:]]+]]

// CHECK: ^[[FIRST]]
// CHECK-NEXT: %[[C1:.*]] = py.constant(#py.str<"text">)
// CHECK-NEXT: py.store %[[C1]] into @foo
// CHECK-NEXT: cf.br ^[[MERGE:[[:alnum:]]+]]
// CHECK-SAME: %[[C1]]

// CHECK: ^[[SECOND]]:
// CHECK-NEXT: %[[C2:.*]] = py.constant(#py.str<"value">)
// CHECK-NEXT: py.store %[[C2]] into @foo
// CHECK-NEXT: cf.br ^[[MERGE]]
// CHECK-SAME: %[[C2]]

// CHECK: ^[[MERGE]]
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: return %[[ARG]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type

py.globalHandle @foo

func private @clobber()

func @test() -> !py.unknown {
    %0 = test.random
    cf.cond_br %0, ^bb0, ^bb1

^bb0:
    call @clobber() : () -> ()
    cf.br ^merge

^bb1:
    %1 = py.constant(#py.str<"value">) : !py.unknown
    py.store %1 into @foo : !py.unknown
    cf.br ^merge

^merge:
    %2 = py.load @foo : !py.unknown
    return %2 : !py.unknown
}

// CHECK-LABEL: func @test
// CHECK: call @clobber
// CHECK-NEXT: cf.br ^[[MERGE:[[:alnum:]]+]]

// CHECK: py.store %{{.*}} into @foo
// CHECK-NEXT: cf.br ^[[MERGE]]

// CHECK: ^[[MERGE]]:
// CHECK-NEXT: %[[RESULT:.*]] = py.load @foo
// CHECK-NEXT: return %[[RESULT]]

// -----

