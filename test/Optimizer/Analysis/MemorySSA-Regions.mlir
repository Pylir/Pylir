// RUN: pylir-opt %s --test-memory-ssa --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type

func.func @test(%arg0 : i1, %length : index) -> index {
    %2 = py.makeList ()
    scf.if %arg0 {
    } else {
        py.list.resize %2 to %length
    }
    %3 = py.list.len %2
    return %3 : index
}

// CHECK-LABEL: memSSA.module
// CHECK-NEXT: %[[ENTRY:.*]] = liveOnEntry
// CHECK-NEXT: %[[DEF:.*]] = def(%[[ENTRY]])
// CHECK-NEXT: py.makeList
// CHECK-NEXT: br ^[[REGION1:.*]], ^[[REGION2:.*]] (), ()
// CHECK-NEXT: ^[[REGION1]]:
// CHECK-NEXT: br ^[[END:.*]] (%[[DEF]])
// CHECK-NEXT: ^[[REGION2]]:
// CHECK-NEXT: %[[DEF2:.*]] = def(%[[DEF]])
// CHECK-NEXT: py.list.resize
// CHECK-NEXT: br ^[[END]] (%[[DEF2]])
// CHECK-NEXT: ^[[END]]
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: use(%[[ARG]])
// CHECK-NEXT: py.list.len
