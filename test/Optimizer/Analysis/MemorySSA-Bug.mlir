// RUN: pylir-opt %s --test-memory-ssa --split-input-file

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.list = #py.type

func private @random() -> i1

func @test4() -> index {
    %0 = py.constant #py.str<"test">
    %1 = py.makeList ()
    br ^condition

^condition: // pred: ^bb0, ^bb2
    %2 = call @random() : () -> i1
    cond_br %2, ^bb1, ^bb5

^bb1: // pred: ^condition
    %3 = call @random() : () -> i1
    cond_br %3, ^bb2, ^bb4

^bb2: // pred: ^bb1
    %unused = py.list.len %1
    br ^condition

^bb4: // pred: ^bb4
    py.list.append %1, %0
    br ^bb5

^bb5: // pred: ^condition, ^bb4
    %5 = py.list.len %1
    return %5 : index
}

// CHECK-LABEL: memSSA.region @test4
// CHECK-NEXT: %[[LIVE_ON_ENTRY:.*]] = memSSA.liveOnEntry
// CHECK-NEXT: %[[DEF1:.*]] = memSSA.def(%[[LIVE_ON_ENTRY]])
// CHECK-NEXT: %[[LIST_DEF:.*]] = py.makeList
// CHECK: memSSA.use(%[[DEF1]])
// CHECK-NEXT: py.list.len %[[LIST_DEF]]
// CHECK: %[[DEF2:.*]] = memSSA.def(%[[DEF1]])
// CHECK-NEXT: py.list.append %[[LIST_DEF]]
// CHECK: ^{{.*}}(%[[MEM_MERGE:.*]]: !memSSA.def)
// CHECK-NEXT: memSSA.use(%[[MEM_MERGE]])
// CHECK-NEXT: py.list.len %[[LIST_DEF]]

func @test5() -> index {
    %0 = py.constant #py.str<"test">
    %1 = py.makeList ()
    br ^back1

^back1: // pred: ^bb0, ^bb2
    br ^back2

^back2: // pred: ^condition
    br ^exit2

^exit2: // pred: ^bb1
    %2 = call @random() : () -> i1
    cond_br %2, ^back2, ^exit1

^exit1: // pred: ^bb4
    %3 = call @random() : () -> i1
    cond_br %3, ^back1, ^bb5

^bb5: // pred: ^condition, ^bb4
    %5 = py.list.len %1
    return %5 : index
}

// CHECK-LABEL: memSSA.region @test5
// CHECK-NEXT: %[[LIVE_ON_ENTRY:.*]] = memSSA.liveOnEntry
// CHECK-NEXT: %[[DEF1:.*]] = memSSA.def(%[[LIVE_ON_ENTRY]])
// CHECK-NEXT: %[[LIST_DEF:.*]] = py.makeList
// CHECK: memSSA.use(%[[DEF1]])
// CHECK-NEXT: py.list.len %[[LIST_DEF]]
