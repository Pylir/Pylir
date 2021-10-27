// RUN: pylir-opt %s | pylir-opt | FileCheck %s
// RUN: pylir-opt %s --mlir-print-op-generic | pylir-opt | FileCheck %s

// CHECK-LABEL: test_constant_integer
func @test_constant_integer() -> !py.dynamic {
    %0 = py.constant #py.int<50>
    return %0 : !py.dynamic
}

// CHECK-LABEL: test_constant_bool
func @test_constant_bool() -> !py.dynamic {
    %0 = py.constant #py.bool<True>
    %1 = py.constant #py.bool<False>
    return %0 : !py.dynamic
}

// CHECK-LABEL: test_constant_float
func @test_constant_float() -> !py.dynamic {
    %0 = py.constant 433.4
    return %0 : !py.dynamic
}

// CHECK-LABEL: test_constant_string
func @test_constant_string() -> !py.dynamic {
    %0 = py.constant "text"
    return %0 : !py.dynamic
}

// CHECK-LABEL: test_constant_list
func @test_constant_list() -> !py.dynamic {
    %0 = py.constant #py.list<[433.4, #py.int<5>]>
    %empty = py.constant #py.list<[]>
    return %0 : !py.dynamic
}

// CHECK-LABEL: test_constant_tuple
func @test_constant_tuple() -> !py.dynamic {
    %0 = py.constant #py.tuple<(433.4, #py.int<5>)>
    %empty = py.constant #py.tuple<()>
    return %0 : !py.dynamic
}

// CHECK-LABEL: test_constant_set
func @test_constant_set() -> !py.dynamic {
    %0 = py.constant #py.set<{433.4, #py.int<5>}>
    %empty = py.constant #py.set<{}>
    return %0 : !py.dynamic
}

// CHECK-LABEL: test_constant_dict
func @test_constant_dict() -> !py.dynamic {
    %0 = py.constant #py.dict<{433.4 to #py.int<5>, "__call__" to #py.int<5>}>
    %empty = py.constant #py.dict<{}>
    return %0 : !py.dynamic
}
