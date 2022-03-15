// RUN: pylir-opt %s -pass-pipeline='builtin.func(load-forwarding)' --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type<slots = {__slots__ = #py.tuple<value = (#py.str<value = "__slots__">)>}>
py.globalValue @builtins.str = #py.type<>
py.globalValue @builtins.tuple = #py.type<>
py.globalValue @foo = #py.type<slots = {__slots__ = #py.tuple<value = (#py.str<value = "test">)>}>

func @test_get_slot() -> !py.dynamic {
    %0 = py.constant @foo
    %1 = py.makeObject %0
    %2 = py.constant #py.str<value = "value">
    py.setSlot "test" of %1 : %0 to %2
    %3 = py.getSlot "test" from %1 : %0
    return %3 : !py.dynamic
}

// CHECK-LABEL: @test_get_slot
// CHECK: %[[C:.*]] = py.constant #py.str<value = "value">
// CHECK: return %[[C]]

// -----

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.str = #py.type<>

func private @bar()

func @test_get_slot_clobbered(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.constant #py.str<value = "value">
    %1 = py.typeOf %arg0
    py.setSlot "test" of %arg0 : %1 to %0
    call @bar() : () -> ()
    %2 = py.getSlot "test" from %arg0 : %1
    return %2 : !py.dynamic
}

// CHECK-LABEL: @test_get_slot_clobbered
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[SLOT:.*]] = py.getSlot "test" from %[[ARG0]]
// CHECK: return %[[SLOT]]

// -----

func @test_get_slot_new_object(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeObject %arg0
    %1 = py.getSlot "test" from %0 : %arg0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @test_get_slot_new_object
// CHECK: %[[C:.*]] = py.constant #py.unbound
// CHECK: return %[[C]]

// -----

func @test_dict_len() -> index {
    %0 = py.makeDict ()
    %1 = py.dict.len %0
    return %1 : index
}

// CHECK-LABEL: @test_dict_len
// CHECK: %[[C:.*]] = arith.constant 0
// CHECK: return %[[C]]

// -----

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.str = #py.type<>

func @test_dict_lookup_setitem(%arg0 : !py.dynamic) -> (!py.dynamic, i1) {
    %0 = py.constant #py.str<value = "value">
    py.dict.setItem %arg0[%0] to %0
    %result, %found = py.dict.tryGetItem %arg0[%0]
    return %result, %found : !py.dynamic, i1
}

// CHECK-LABEL: @test_dict_lookup_setitem
// CHECK-DAG: %[[C:.*]] = py.constant #py.str<value = "value">
// CHECK-DAG: %[[TRUE:.*]] = arith.constant true
// CHECK: return %[[C]], %[[TRUE]]

// -----

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.str = #py.type<>

func @test_dict_lookup_delitem(%arg0 : !py.dynamic) -> (!py.dynamic, i1) {
    %0 = py.constant #py.str<value = "value">
    py.dict.delItem %0 from %arg0
    %result, %found = py.dict.tryGetItem %arg0[%0]
    return %result, %found : !py.dynamic, i1
}

// CHECK-LABEL: @test_dict_lookup_delitem
// CHECK-DAG: %[[C:.*]] = py.constant #py.unbound
// CHECK-DAG: %[[FALSE:.*]] = arith.constant false
// CHECK: return %[[C]], %[[FALSE]]

// -----

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.str = #py.type<>

func @test_dict_lookup_makeDict() -> (!py.dynamic, i1) {
    %0 = py.constant #py.str<value = "value">
    %1 = py.makeDict ()
    %result, %found = py.dict.tryGetItem %1[%0]
    return %result, %found : !py.dynamic, i1
}

// CHECK-LABEL: @test_dict_lookup_makeDict
// CHECK-DAG: %[[C:.*]] = py.constant #py.unbound
// CHECK-DAG: %[[FALSE:.*]] = arith.constant false
// CHECK: return %[[C]], %[[FALSE]]

// -----

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.str = #py.type<>

func @test_list_len() -> index {
    %0 = py.constant #py.str<value = "value">
    %1 = py.makeDict ()
    %2 = py.makeList (%0, %1)
    %3 = py.list.len %2
    return %3 : index
}

// CHECK-LABEL: @test_list_len
// CHECK: %[[C:.*]] = arith.constant 2
// CHECK: return %[[C]]
