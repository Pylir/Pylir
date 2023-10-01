// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int

py.func @test() -> !py.dynamic {
    %0 = arith.constant 5 : index
    %1 = int_fromUnsigned %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-NEXT: %[[C:.*]] = constant(#py.int<5>)
// CHECK-NEXT: return %[[C]]

py.func @test2() -> !py.dynamic {
    %0 = arith.constant -5 : index
    %1 = int_fromUnsigned %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @test2
// CHECK-NEXT: %[[C:.*]] = constant(#py.int<{{[0-9]+}}>)
// CHECK-NEXT: return %[[C]]
