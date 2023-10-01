// RUN: pylir-opt %s --test-memory-ssa --split-input-file | FileCheck %s

py.func @test() -> index {
    %0 = constant(#py.str<"test">)
    %hash = str_hash %0
    %1 = makeDict ()
    %2 = makeDict ()
    dict_setItem %1[%0 hash(%hash)] to %0
    dict_setItem %2[%0 hash(%hash)] to %0
    %3 = dict_len %1
    return %3 : index
}

// CHECK-LABEL: memSSA.module
// CHECK-NEXT: %[[LIVE_ON_ENTRY:.*]] = liveOnEntry
// CHECK-NEXT: %[[DEF_OBJECT:.*]] = def(%[[LIVE_ON_ENTRY:.*]])
// CHECK-NEXT: %[[DICT1:.*]] = py.makeDict ()
// CHECK-NEXT: %[[DEF_DICT:.*]] = def(%[[LIVE_ON_ENTRY:.*]])
// CHECK-NEXT: %[[DICT1]] = py.makeDict ()
// CHECK-NEXT: %[[DEF_OBJECT2:.*]] = def(%[[DEF_OBJECT]])
// CHECK-NEXT: %[[DICT2:.*]] = py.makeDict ()
// CHECK-NEXT: %[[DEF_DICT2:.*]] = def(%[[DEF_DICT]])
// CHECK-NEXT: %[[DICT2]] = py.makeDict ()
// CHECK-NEXT: %[[DEF3:.*]] = def(%[[DEF_DICT2]])
// CHECK-NEXT: dict_setItem %[[DICT1]]
// CHECK-NEXT: %[[DEF4:.*]] = def(%[[DEF3]])
// CHECK-NEXT: dict_setItem %[[DICT2]]
// CHECK-NEXT: use(%[[DEF3]])
// CHECK-NEXT: dict_len %[[DICT1]]
