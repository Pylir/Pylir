// RUN: pylir-opt %s --test-memory-ssa --split-input-file | FileCheck %s

py.func @test(%arg0 : i1, %length : index) -> index {
    %2 = makeList ()
    scf.if %arg0 {
    } else {
        py.list_resize %2 to %length
    }
    %3 = list_len %2
    return %3 : index
}

// CHECK-LABEL: memSSA.module
// CHECK-NEXT: %[[ENTRY:.*]] = liveOnEntry
// CHECK-NEXT: %[[DEF_OBJECT:.*]] = def(%[[ENTRY]])
// CHECK-NEXT: makeList
// CHECK-NEXT: %[[DEF_LIST:.*]] = def(%[[ENTRY]])
// CHECK-NEXT: makeList
// CHECK-NEXT: br ^[[REGION1:.*]], ^[[REGION2:.*]] (), ()
// CHECK-NEXT: ^[[REGION1]]:
// CHECK-NEXT: br ^[[END:.*]] (%[[DEF_LIST]])
// CHECK-NEXT: ^[[REGION2]]:
// CHECK-NEXT: %[[DEF2:.*]] = def(%[[DEF_LIST]])
// CHECK-NEXT: py.list_resize
// CHECK-NEXT: br ^[[END]] (%[[DEF2]])
// CHECK-NEXT: ^[[END]]
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: use(%[[ARG]])
// CHECK-NEXT: list_len
