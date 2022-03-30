// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue const @tuple = #py.tuple<value = (#py.str<value = "__slots__">)>
py.globalValue const @builtins.type = #py.type<slots = {__slots__ = @tuple}>
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.str = #py.type

func @test() -> !py.unknown {
    %0 = py.constant(@builtins.type) : !py.unknown
    %1 = py.getSlot "__slots__" from %0 : %0 : (!py.unknown, !py.unknown) -> !py.unknown
    return %1 : !py.unknown
}

// CHECK-LABEL: func @test
// CHECK-NEXT: %[[C:.*]] = py.constant(@tuple)
// CHECK-NEXT: return %[[C]]
