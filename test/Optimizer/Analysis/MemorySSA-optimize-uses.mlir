// RUN: pylir-opt %s --test-memory-ssa --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.dict = #py.type
py.globalValue @builtins.str = #py.type

func.func @test() -> index {
    %0 = py.constant(#py.str<"test">)
    %hash = py.str.hash %0
    %1 = py.makeDict ()
    %2 = py.makeDict ()
    py.dict.setItem %1[%0 hash(%hash)] to %0
    py.dict.setItem %2[%0 hash(%hash)] to %0
    %3 = py.dict.len %1
    return %3 : index
}

// CHECK-LABEL: memSSA.module
// CHECK-NEXT: %[[LIVE_ON_ENTRY:.*]] = liveOnEntry
// CHECK-NEXT: %[[DEF:.*]] = def(%[[LIVE_ON_ENTRY:.*]])
// CHECK-NEXT: %[[DICT1:.*]] = py.makeDict ()
// CHECK-NEXT: %[[DEF2:.*]] = def(%[[DEF]])
// CHECK-NEXT: %[[DICT2:.*]] = py.makeDict ()
// CHECK-NEXT: %[[DEF3:.*]] = def(%[[DEF2]])
// CHECK-NEXT: py.dict.setItem %[[DICT1]]
// CHECK-NEXT: %[[DEF4:.*]] = def(%[[DEF3]])
// CHECK-NEXT: py.dict.setItem %[[DICT2]]
// CHECK-NEXT: use(%[[DEF3]])
// CHECK-NEXT: py.dict.len %[[DICT1]]
