// RUN: pylir-opt %s -pass-pipeline='func.func(load-forwarding)' --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type<slots = {__slots__ = #py.tuple<value = (#py.str<value = "__slots__">)>}>
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @foo = #py.type<slots = {__slots__ = #py.tuple<value = (#py.str<value = "test">)>}>

func @test_get_slot() -> !py.unknown {
    %0 = py.constant(@foo) : !py.unknown
    %1 = py.makeObject %0 : (!py.unknown) -> !py.unknown
    %2 = py.constant(#py.str<value = "value">) : !py.unknown
    py.setSlot "test" of %1 : %0 to %2 : !py.unknown, !py.unknown, !py.unknown
    %3 = py.getSlot "test" from %1 : %0 : (!py.unknown, !py.unknown) -> !py.unknown
    return %3 : !py.unknown
}

// CHECK-LABEL: @test_get_slot
// CHECK: %[[C:.*]] = py.constant(#py.str<value = "value">)
// CHECK: return %[[C]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type

func private @bar()

func @test_get_slot_clobbered(%arg0 : !py.unknown) -> !py.unknown {
    %0 = py.constant(#py.str<value = "value">) : !py.unknown
    %1 = py.typeOf %arg0 : (!py.unknown) -> !py.unknown
    py.setSlot "test" of %arg0 : %1 to %0 : !py.unknown, !py.unknown, !py.unknown
    call @bar() : () -> ()
    %2 = py.getSlot "test" from %arg0 : %1 : (!py.unknown, !py.unknown) -> !py.unknown
    return %2 : !py.unknown
}

// CHECK-LABEL: @test_get_slot_clobbered
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[SLOT:.*]] = py.getSlot "test" from %[[ARG0]]
// CHECK: return %[[SLOT]]

// -----

func @test_get_slot_new_object(%arg0 : !py.unknown) -> !py.unknown {
    %0 = py.makeObject %arg0 : (!py.unknown) -> !py.unknown
    %1 = py.getSlot "test" from %0 : %arg0 : (!py.unknown, !py.unknown) -> !py.unknown
    return %1 : !py.unknown
}

// CHECK-LABEL: @test_get_slot_new_object
// CHECK: %[[C:.*]] = py.constant(#py.unbound)
// CHECK: return %[[C]]

// -----

func @test_dict_len() -> index {
    %0 = py.makeDict ()
    %1 = py.dict.len %0 : !py.class<@builtins.dict>
    return %1 : index
}

// CHECK-LABEL: @test_dict_len
// CHECK: %[[C:.*]] = arith.constant 0
// CHECK: return %[[C]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type

func @test_dict_lookup_setitem(%arg0 : !py.unknown) -> (!py.unknown, i1) {
    %0 = py.constant(#py.str<value = "value">) : !py.unknown
    py.dict.setItem %arg0[%0] to %0 : !py.unknown, !py.unknown, !py.unknown
    %result, %found = py.dict.tryGetItem %arg0[%0] : (!py.unknown, !py.unknown) -> !py.unknown
    return %result, %found : !py.unknown, i1
}

// CHECK-LABEL: @test_dict_lookup_setitem
// CHECK-DAG: %[[C:.*]] = py.constant(#py.str<value = "value">)
// CHECK-DAG: %[[TRUE:.*]] = arith.constant true
// CHECK: return %[[C]], %[[TRUE]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type

func @test_dict_lookup_delitem(%arg0 : !py.unknown) -> (!py.unknown, i1) {
    %0 = py.constant(#py.str<value = "value">) : !py.unknown
    py.dict.delItem %0 from %arg0 : !py.unknown, !py.unknown
    %result, %found = py.dict.tryGetItem %arg0[%0] : (!py.unknown, !py.unknown) -> !py.unknown
    return %result, %found : !py.unknown, i1
}

// CHECK-LABEL: @test_dict_lookup_delitem
// CHECK-DAG: %[[C:.*]] = py.constant(#py.unbound)
// CHECK-DAG: %[[FALSE:.*]] = arith.constant false
// CHECK: return %[[C]], %[[FALSE]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type

func @test_dict_lookup_makeDict() -> (!py.unknown, i1) {
    %0 = py.constant(#py.str<value = "value">) : !py.unknown
    %1 = py.makeDict ()
    %result, %found = py.dict.tryGetItem %1[%0] : (!py.class<@builtins.dict>, !py.unknown) -> !py.unknown
    return %result, %found : !py.unknown, i1
}

// CHECK-LABEL: @test_dict_lookup_makeDict
// CHECK-DAG: %[[C:.*]] = py.constant(#py.unbound)
// CHECK-DAG: %[[FALSE:.*]] = arith.constant false
// CHECK: return %[[C]], %[[FALSE]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type

func @test_list_len() -> index {
    %0 = py.constant(#py.str<value = "value">) : !py.unknown
    %1 = py.makeDict ()
    %2 = py.makeList (%0, %1) : !py.unknown, !py.class<@builtins.dict>
    %3 = py.list.len %2 : !py.class<@builtins.list>
    return %3 : index
}

// CHECK-LABEL: @test_list_len
// CHECK: %[[C:.*]] = arith.constant 2
// CHECK: return %[[C]]
