// RUN: pylir-opt %s -pass-pipeline='builtin.module(any(pylir-load-forwarding))' --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type<slots = {__slots__ = #py.tuple<(#py.str<"__slots__">)>}>
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @foo = #py.type<slots = {instance_slots = #py.tuple<(#py.str<"test">)>}>

func.func @test_get_slot() -> !py.dynamic {
    %0 = py.constant(#py.ref<@foo>)
    %1 = py.makeObject %0
    %2 = py.constant(#py.str<"value">)
    %c0 = arith.constant 0 : index
    py.setSlot %1[%c0] to %2
    %3 = py.getSlot %1[%c0]
    return %3 : !py.dynamic
}

// CHECK-LABEL: @test_get_slot
// CHECK: %[[C:.*]] = py.constant(#py.str<"value">)
// CHECK: return %[[C]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type

func.func private @bar()

func.func @test_get_slot_clobbered(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.constant(#py.str<"value">)
    %c0 = arith.constant 0 : index
    py.setSlot %arg0[%c0] to %0
    call @bar() : () -> ()
    %2 = py.getSlot %arg0[%c0]
    return %2 : !py.dynamic
}

// CHECK-LABEL: @test_get_slot_clobbered
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[SLOT:.*]] = py.getSlot %[[ARG0]]
// CHECK: return %[[SLOT]]

// -----

func.func @test_get_slot_new_object(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeObject %arg0
    %c0 = arith.constant 0 : index
    %1 = py.getSlot %0[%c0]
    return %1 : !py.dynamic
}

// CHECK-LABEL: @test_get_slot_new_object
// CHECK: %[[C:.*]] = py.constant(#py.unbound)
// CHECK: return %[[C]]

// -----

func.func @test_dict_len() -> index {
    %0 = py.makeDict ()
    %1 = py.dict.len %0
    return %1 : index
}

// CHECK-LABEL: @test_dict_len
// CHECK: %[[C:.*]] = arith.constant 0
// CHECK: return %[[C]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type

func.func @test_dict_lookup_setitem(%arg0 : !py.dynamic, %hash : index) -> !py.dynamic {
    %0 = py.constant(#py.str<"value">)
    py.dict.setItem %arg0[%0 hash(%hash)] to %0
    %result = py.dict.tryGetItem %arg0[%0 hash(%hash)]
    return %result : !py.dynamic
}

// CHECK-LABEL: @test_dict_lookup_setitem
// CHECK-DAG: %[[C:.*]] = py.constant(#py.str<"value">)
// CHECK: return %[[C]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type

func.func @test_dict_lookup_delitem(%arg0 : !py.dynamic, %hash: index) -> !py.dynamic {
    %0 = py.constant(#py.str<"value">)
    py.dict.delItem %0 hash(%hash) from %arg0
    %result = py.dict.tryGetItem %arg0[%0 hash(%hash)]
    return %result : !py.dynamic
}

// CHECK-LABEL: @test_dict_lookup_delitem
// CHECK-DAG: %[[C:.*]] = py.constant(#py.unbound)
// CHECK: return %[[C]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type

func.func @test_dict_lookup_makeDict(%hash : index) -> (!py.dynamic, !py.dynamic) {
    %0 = py.constant(#py.str<"value">)
    %1 = py.makeDict ()
    %result = py.dict.tryGetItem %1[%0 hash(%hash)]
    %2 = py.makeDict (%0 hash(%hash) : %0)
    %res2 = py.dict.tryGetItem %2[%0 hash(%hash)]
    return %result, %res2 : !py.dynamic, !py.dynamic
}

// CHECK-LABEL: @test_dict_lookup_makeDict
// CHECK-DAG: %[[U:.*]] = py.constant(#py.unbound)
// CHECK-DAG: %[[C:.*]] = py.constant(#py.str<"value">)
// CHECK: return %[[U]], %[[C]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type

func.func @test_list_len() -> index {
    %0 = py.constant(#py.str<"value">)
    %1 = py.makeDict ()
    %2 = py.makeList (%0, %1)
    %3 = py.list.len %2
    return %3 : index
}

// CHECK-LABEL: @test_list_len
// CHECK: %[[C:.*]] = arith.constant 2
// CHECK: return %[[C]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type

func.func @test_resources(%arg0 : !py.dynamic) -> index {
    %0 = arith.constant 5 : index
    py.list.resize %arg0 to %0
    %1 = py.typeOf %arg0
    %2 = py.constant(#py.str<"mhm">)
    %c0 = arith.constant 0 : index
    py.setSlot %arg0[%c0] to %2
    %3 = py.list.len %arg0
    return %3 : index
}

// CHECK-LABEL: @test_resources
// CHECK: %[[C:.*]] = arith.constant 5
// CHECK-NOT: py.list.len
// CHECK: return %[[C]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.float = #py.type

func.func @test_dict_lookup_makeDict_equal(%hash : index) -> !py.dynamic {
    %0 = py.constant(#py.int<5>)
    %1 = py.constant(#py.float<5.0>)
    %2 = py.makeDict (%0 hash(%hash) : %0)
    %res2 = py.dict.tryGetItem %2[%1 hash(%hash)]
    return %res2 : !py.dynamic
}

// CHECK-LABEL: @test_dict_lookup_makeDict_equal
// CHECK: %[[C:.*]] = py.constant(#py.int<5>)
// CHECK: return %[[C]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.float = #py.type

func.func @test_dict_lookup_makeDict_not_equal(%hash : index) -> !py.dynamic {
    %0 = py.constant(#py.int<5>)
    %1 = py.constant(#py.float<3.0>)
    %2 = py.makeDict (%0 hash(%hash) : %0)
    %res2 = py.dict.tryGetItem %2[%1 hash(%hash)]
    return %res2 : !py.dynamic
}

// CHECK-LABEL: @test_dict_lookup_makeDict_not_equal
// CHECK: %[[C:.*]] = py.constant(#py.unbound)
// CHECK: return %[[C]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.float = #py.type

func.func @test_dict_lookup_makeDict_neg(%hash : index, %key : !py.dynamic) -> !py.dynamic {
    %0 = py.constant(#py.int<5>)
    %2 = py.makeDict (%key hash(%hash) : %0)
    %res2 = py.dict.tryGetItem %2[%0 hash(%hash)]
    return %res2 : !py.dynamic
}

// CHECK-LABEL: @test_dict_lookup_makeDict_neg
// CHECK: %[[D:.*]] = py.makeDict
// CHECK-NEXT: %[[L:.*]] = py.dict.tryGetItem
// CHECK-NEXT: return %[[L]]
