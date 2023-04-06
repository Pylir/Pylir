// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type<instance_slots = <(#py.str<"first">, #py.str<"second">)>>
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.str = #py.type

py.func @test() -> !py.dynamic {
    %0 = constant(#py.ref<@builtins.type>)
    %1 = type_slots %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-NEXT: %[[C:.*]] = constant(#py.tuple<(#py.str<"first">, #py.str<"second">)>)
// CHECK-NEXT: return %[[C]]
