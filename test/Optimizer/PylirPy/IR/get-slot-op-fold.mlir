// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue const @foo = #py.tuple<(#py.str<"__slots__">)>
py.globalValue const @builtins.type = #py.type<instance_slots = <(#py.str<"__slots__">)>, slots = { __slots__ = #py.ref<@foo> }>
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.str = #py.type

py.func @test() -> !py.dynamic {
    %0 = constant(#py.ref<@builtins.type>)
    %c0 = arith.constant 0 : index
    %1 = getSlot %0[%c0]
    return %1 : !py.dynamic
}

// CHECK-LABEL: func @test
// CHECK-NEXT: %[[C:.*]] = constant(#py.ref<@foo>)
// CHECK-NEXT: return %[[C]]
