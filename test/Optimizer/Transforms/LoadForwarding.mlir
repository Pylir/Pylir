// RUN: pylir-opt %s -pass-pipeline='builtin.module(any(pylir-load-forwarding))' --split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type<slots = {__slots__ = #py.tuple<(#py.str<"__slots__">)>}>>
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#foo = #py.globalValue<foo, initializer = #py.type<slots = {instance_slots = #py.tuple<(#py.str<"test">)>}>>

py.func @test_get_slot() -> !py.dynamic {
    %0 = constant(#foo)
    %1 = makeObject %0
    %2 = constant(#py.str<"value">)
    %c0 = arith.constant 0 : index
    setSlot %1[%c0] to %2
    %3 = getSlot %1[%c0]
    return %3 : !py.dynamic
}

// CHECK-LABEL: @test_get_slot
// CHECK: %[[C:.*]] = constant(#py.str<"value">)
// CHECK: return %[[C]]

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str

py.func private @bar()

py.func @test_get_slot_clobbered(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = constant(#py.str<"value">)
    %c0 = arith.constant 0 : index
    setSlot %arg0[%c0] to %0
    call @bar() : () -> ()
    %2 = getSlot %arg0[%c0]
    return %2 : !py.dynamic
}

// CHECK-LABEL: @test_get_slot_clobbered
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[SLOT:.*]] = getSlot %[[ARG0]]
// CHECK: return %[[SLOT]]

// -----

py.func @test_get_slot_new_object(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = makeObject %arg0
    %c0 = arith.constant 0 : index
    %1 = getSlot %0[%c0]
    return %1 : !py.dynamic
}

// CHECK-LABEL: @test_get_slot_new_object
// CHECK: %[[C:.*]] = constant(#py.unbound)
// CHECK: return %[[C]]

// -----

py.func @test_dict_len() -> index {
    %0 = makeDict ()
    %1 = dict_len %0
    return %1 : index
}

// CHECK-LABEL: @test_dict_len
// CHECK: %[[C:.*]] = arith.constant 0
// CHECK: return %[[C]]

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str

py.func @test_dict_lookup_setitem(%arg0 : !py.dynamic, %hash : index) -> !py.dynamic {
    %0 = constant(#py.str<"value">)
    dict_setItem %arg0[%0 hash(%hash)] to %0
    %result = dict_tryGetItem %arg0[%0 hash(%hash)]
    return %result : !py.dynamic
}

// CHECK-LABEL: @test_dict_lookup_setitem
// CHECK-DAG: %[[C:.*]] = constant(#py.str<"value">)
// CHECK: return %[[C]]

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str

py.func @test_dict_lookup_delitem(%arg0 : !py.dynamic, %hash: index) -> !py.dynamic {
    %0 = constant(#py.str<"value">)
    dict_delItem %0 hash(%hash) from %arg0
    %result = dict_tryGetItem %arg0[%0 hash(%hash)]
    return %result : !py.dynamic
}

// CHECK-LABEL: @test_dict_lookup_delitem
// CHECK-DAG: %[[C:.*]] = constant(#py.unbound)
// CHECK: return %[[C]]

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str

py.func @test_dict_lookup_makeDict(%hash : index) -> (!py.dynamic, !py.dynamic) {
    %0 = constant(#py.str<"value">)
    %1 = makeDict ()
    %result = dict_tryGetItem %1[%0 hash(%hash)]
    %2 = makeDict (%0 hash(%hash) : %0)
    %res2 = dict_tryGetItem %2[%0 hash(%hash)]
    return %result, %res2 : !py.dynamic, !py.dynamic
}

// CHECK-LABEL: @test_dict_lookup_makeDict
// CHECK-DAG: %[[U:.*]] = constant(#py.unbound)
// CHECK-DAG: %[[C:.*]] = constant(#py.str<"value">)
// CHECK: return %[[U]], %[[C]]

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str

py.func @test_list_len() -> index {
    %0 = constant(#py.str<"value">)
    %1 = makeDict ()
    %2 = makeList (%0, %1)
    %3 = list_len %2
    return %3 : index
}

// CHECK-LABEL: @test_list_len
// CHECK: %[[C:.*]] = arith.constant 2
// CHECK: return %[[C]]

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str

py.func @test_resources(%arg0 : !py.dynamic) -> index {
    %0 = arith.constant 5 : index
    list_resize %arg0 to %0
    %1 = typeOf %arg0
    %2 = constant(#py.str<"mhm">)
    %c0 = arith.constant 0 : index
    setSlot %arg0[%c0] to %2
    %3 = list_len %arg0
    return %3 : index
}

// CHECK-LABEL: @test_resources
// CHECK: %[[C:.*]] = arith.constant 5
// CHECK-NOT: list_len
// CHECK: return %[[C]]

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int
#builtins_float = #py.globalValue<builtins.float, initializer = #py.type>
py.external @builtins.float, #builtins_float

py.func @test_dict_lookup_makeDict_equal(%hash : index) -> !py.dynamic {
    %0 = constant(#py.int<5>)
    %1 = constant(#py.float<5.0>)
    %2 = makeDict (%0 hash(%hash) : %0)
    %res2 = dict_tryGetItem %2[%1 hash(%hash)]
    return %res2 : !py.dynamic
}

// CHECK-LABEL: @test_dict_lookup_makeDict_equal
// CHECK: %[[C:.*]] = constant(#py.int<5>)
// CHECK: return %[[C]]

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int
#builtins_float = #py.globalValue<builtins.float, initializer = #py.type>
py.external @builtins.float, #builtins_float

py.func @test_dict_lookup_makeDict_not_equal(%hash : index) -> !py.dynamic {
    %0 = constant(#py.int<5>)
    %1 = constant(#py.float<3.0>)
    %2 = makeDict (%0 hash(%hash) : %0)
    %res2 = dict_tryGetItem %2[%1 hash(%hash)]
    return %res2 : !py.dynamic
}

// CHECK-LABEL: @test_dict_lookup_makeDict_not_equal
// CHECK: %[[C:.*]] = constant(#py.unbound)
// CHECK: return %[[C]]

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int
#builtins_float = #py.globalValue<builtins.float, initializer = #py.type>
py.external @builtins.float, #builtins_float

py.func @test_dict_lookup_makeDict_neg(%hash : index, %key : !py.dynamic) -> !py.dynamic {
    %0 = constant(#py.int<5>)
    %2 = makeDict (%key hash(%hash) : %0)
    %res2 = dict_tryGetItem %2[%0 hash(%hash)]
    return %res2 : !py.dynamic
}

// CHECK-LABEL: @test_dict_lookup_makeDict_neg
// CHECK: %[[D:.*]] = makeDict
// CHECK-NEXT: %[[L:.*]] = dict_tryGetItem
// CHECK-NEXT: return %[[L]]
