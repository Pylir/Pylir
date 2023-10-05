// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

#foo = #py.globalValue<foo, const, initializer = #py.tuple<(#py.str<"__slots__">)>>
#builtins_type = #py.globalValue<builtins.type, const, initializer = #py.type<instance_slots = <(#py.str<"__slots__">)>, slots = { __slots__ = #foo }>>

py.func @test() -> !py.dynamic {
    %0 = constant(#builtins_type)
    %c0 = arith.constant 0 : index
    %1 = getSlot %0[%c0]
    return %1 : !py.dynamic
}

// CHECK: #[[$FOO:.*]] = #py.globalValue<foo{{.*}}>

// CHECK-LABEL: func @test
// CHECK-NEXT: %[[C:.*]] = constant(#[[$FOO]])
// CHECK-NEXT: return %[[C]]

// -----

// CHECK-LABEL: func @unknown_type
py.func @unknown_type() -> !py.dynamic {
    // CHECK: %[[C:.*]] = constant(#py.obj<#{{.*}}>)
    // CHECK: %[[SLOT:.*]] = getSlot %[[C]]
    // CHECK: return %[[SLOT]]

    %0 = constant(#py.obj<#py.globalValue<imported>>)
    %c0 = arith.constant 0 : index
    %1 = getSlot %0[%c0]
    return %1 : !py.dynamic
}
