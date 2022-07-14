// RUN: pylir-opt %s --test-memory-ssa --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type

func.func @test(%length : index) -> index {
    %1 = py.makeList ()
    py.list.resize %1 to %length
    %2 = py.list.len %1
    return %2 : index
}

// CHECK-LABEL: memSSA.module
// CHECK-NEXT: %[[LIVE_ON_ENTRY:.*]] = liveOnEntry
// CHECK-NEXT: %[[DEF:.*]] = def(%[[LIVE_ON_ENTRY]])
// CHECK-NEXT: // {{.*}} py.makeList
// CHECK-NEXT: %[[APPEND:.*]] = def(%[[DEF]])
// CHECK-NEXT: // py.list.resize
// CHECK-NEXT: use(%[[APPEND]])
// CHECK-NEXT: // {{.*}} py.list.len

func.func @test2(%arg0 : i1, %length : index) -> index {
    %1 = py.makeList ()
    cf.cond_br %arg0, ^bb1, ^bb2

^bb1:
    py.list.resize %1 to %length
    cf.br ^bb2

^bb2:
    %2 = py.list.len %1
    return %2 : index
}

// CHECK-LABEL: memSSA.module
// CHECK-NEXT: %[[LIVE_ON_ENTRY:.*]] = liveOnEntry
// CHECK-NEXT: %[[DEF:.*]] = def(%[[LIVE_ON_ENTRY]])
// CHECK-NEXT: // {{.*}} py.makeList
// CHECK-NEXT: br ^[[FIRST:.*]], ^[[SECOND:.*]] (), (%[[DEF]])
// CHECK-NEXT: ^[[FIRST]]:
// CHECK-NEXT: %[[APPEND:.*]] = def(%[[DEF]])
// CHECK-NEXT: // py.list.resize
// CHECK-NEXT: br ^[[SECOND]] (%[[APPEND]])
// CHECK-NEXT: ^[[SECOND]]
// CHECK-SAME: %[[MERGE:[[:alnum:]]+]]
// CHECK-NEXT: use(%[[MERGE]])
// CHECK-NEXT: // {{.*}} py.list.len

func.func @test3(%length : index) -> index {
    %1 = py.makeList ()
    cf.br ^condition

^condition:
    %2 = test.random
    cf.cond_br %2, ^bb1, ^bb2

^bb1:
    py.list.resize %1 to %length
    cf.br ^condition

^bb2:
    %3 = py.list.len %1
    return %3 : index
}

// CHECK-LABEL: memSSA.module
// CHECK-NEXT: %[[LIVE_ON_ENTRY:.*]] = liveOnEntry
// CHECK-NEXT: %[[DEF:.*]] = def(%[[LIVE_ON_ENTRY]])
// CHECK-NEXT: // {{.*}} py.makeList
// CHECK-NEXT: br ^[[FIRST:.*]] (%[[DEF]])
// CHECK-NEXT: ^[[FIRST]]
// CHECK-SAME: %[[COND:[[:alnum:]]+]]
// CHECK-NEXT: br ^[[BODY:.*]], ^[[EXIT:.*]] (), ()
// CHECK-NEXT: ^[[BODY]]:
// CHECK-NEXT: %[[NEW_DEF:.*]] = def(%[[COND]])
// CHECK-NEXT: // py.list.resize
// CHECK-NEXT: br ^[[FIRST]] (%[[NEW_DEF]])
// CHECK-NEXT: ^[[EXIT]]:
// CHECK-NEXT: use(%[[COND]])
// CHECK-NEXT: // {{.*}} py.list.len

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type

func.func private @bar()

func.func @test4(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.constant(#py.str<"value">)
    %1 = py.typeOf %arg0
    py.setSlot "test" of %arg0 : %1 to %0
    call @bar() : () -> ()
    %2 = py.getSlot "test" from %arg0 : %1
    return %2 : !py.dynamic
}

// skip @bar
// CHECK-LABEL: memSSA.module

// CHECK-LABEL: memSSA.module
// CHECK-NEXT: %[[LIVE_ON_ENTRY:.*]] = liveOnEntry
// CHECK-NEXT: %[[DEF:.*]] = def(%[[LIVE_ON_ENTRY]])
// CHECK-NEXT: py.setSlot "test"
// CHECK-NEXT: %[[DEF2:.*]] = def(%[[DEF]])
// CHECK-NEXT: call @bar()
// CHECK-NEXT: use(%[[DEF2]])
// CHECK-NEXT: py.getSlot "test"
