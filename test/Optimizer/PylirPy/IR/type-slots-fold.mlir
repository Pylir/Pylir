// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, const, initializer = #py.type<instance_slots = <(#py.str<"first">, #py.str<"second">)>>>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, const, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str

py.func @test() -> !py.dynamic {
    %0 = constant(#builtins_type)
    %1 = type_slots %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-NEXT: %[[C:.*]] = constant(#py.tuple<(#py.str<"first">, #py.str<"second">)>)
// CHECK-NEXT: return %[[C]]
