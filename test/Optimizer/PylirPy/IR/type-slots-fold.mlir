// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type<instance_slots = <(#py.str<"first">, #py.str<"second">)>>
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.str = #py.type

func.func @test() -> !py.dynamic {
    %0 = py.constant(#py.ref<@builtins.type>)
    %1 = py.type.slots %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-NEXT: %[[C:.*]] = py.constant(#py.tuple<(#py.str<"first">, #py.str<"second">)>)
// CHECK-NEXT: return %[[C]]
