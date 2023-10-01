// RUN: pylir-opt %s --test-memory-ssa --split-input-file

py.func @test4(%length : index) -> index {
    %1 = makeList ()
    cf.br ^condition

^condition: // pred: ^bb0, ^bb2
    %2 = test.random
    cf.cond_br %2, ^bb1, ^bb5

^bb1: // pred: ^condition
    %3 = test.random
    cf.cond_br %3, ^bb2, ^bb4

^bb2: // pred: ^bb1
    %unused = list_len %1
    cf.br ^condition

^bb4: // pred: ^bb4
    list_resize %1 to %length
    cf.br ^bb5

^bb5: // pred: ^condition, ^bb4
    %5 = list_len %1
    return %5 : index
}

// CHECK-LABEL: memSSA.module @test4
// CHECK-NEXT: %[[LIVE_ON_ENTRY:.*]] = liveOnEntry
// CHECK-NEXT: %[[DEF1:.*]] = def(%[[LIVE_ON_ENTRY]])
// CHECK-NEXT: %[[LIST_DEF:.*]] = makeList
// CHECK: use(%[[DEF1]])
// CHECK-NEXT: list_len %[[LIST_DEF]]
// CHECK: %[[DEF2:.*]] = def(%[[DEF1]])
// CHECK-NEXT: list_resize %[[LIST_DEF]]
// CHECK: ^{{.*}}(%[[MEM_MERGE:.*]]: !def)
// CHECK-NEXT: use(%[[MEM_MERGE]])
// CHECK-NEXT: list_len %[[LIST_DEF]]

py.func @test5() -> index {
    %1 = makeList ()
    cf.br ^back1

^back1: // pred: ^bb0, ^bb2
    cf.br ^back2

^back2: // pred: ^condition
    cf.br ^exit2

^exit2: // pred: ^bb1
    %2 = test.random
    cf.cond_br %2, ^back2, ^exit1

^exit1: // pred: ^bb4
    %3 = test.random
    cf.cond_br %3, ^back1, ^bb5

^bb5: // pred: ^condition, ^bb4
    %5 = list_len %1
    return %5 : index
}

// CHECK-LABEL: memSSA.module @test5
// CHECK-NEXT: %[[LIVE_ON_ENTRY:.*]] = liveOnEntry
// CHECK-NEXT: %[[DEF1:.*]] = def(%[[LIVE_ON_ENTRY]])
// CHECK-NEXT: %[[LIST_DEF:.*]] = makeList
// CHECK: use(%[[DEF1]])
// CHECK-NEXT: list_len %[[LIST_DEF]]

py.func @test6() {
    %0 = constant(#py.str<"test">)
    %1 = makeList ()
    cf.br ^bb2

^exit1:
    return

^bb2:
    %2 = test.random
    cf.cond_br %2, ^exit1, ^exit2

^exit2:
    return
}

// CHECK-LABEL: memSSA.module @test6
