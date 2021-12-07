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
    %0 = py.constant #py.float<433.4>
    return %0 : !py.dynamic
}

// CHECK-LABEL: test_constant_string
func @test_constant_string() -> !py.dynamic {
    %0 = py.constant #py.str<"text">
    return %0 : !py.dynamic
}

// CHECK-LABEL: test_constant_list
func @test_constant_list() -> !py.dynamic {
    %0 = py.constant #py.list<[#py.float<433.4>, #py.int<5>]>
    %empty = py.constant #py.list<[]>
    return %0 : !py.dynamic
}

// CHECK-LABEL: test_constant_tuple
func @test_constant_tuple() -> !py.dynamic {
    %0 = py.constant #py.tuple<(#py.float<433.4>, #py.int<5>)>
    %empty = py.constant #py.tuple<()>
    return %0 : !py.dynamic
}

// CHECK-LABEL: test_constant_set
func @test_constant_set() -> !py.dynamic {
    %0 = py.constant #py.set<{#py.float<433.4>, #py.int<5>}>
    %empty = py.constant #py.set<{}>
    return %0 : !py.dynamic
}

// CHECK-LABEL: test_constant_dict
func @test_constant_dict() -> !py.dynamic {
    %0 = py.constant #py.dict<{#py.float<433.4> to #py.int<5>, #py.str<"__call__"> to #py.int<5>}>
    %empty = py.constant #py.dict<{}>
    return %0 : !py.dynamic
}

// CHECK-LABEL: test_objects
func @test_objects() -> !py.dynamic {
    %0 = py.constant #py.obj<type: @a>
    %1 = py.constant #py.obj<type: @a, value: #py.int<1>>
    %2 = py.constant #py.obj<type: @a, value: #py.int<1>, slots: #py.slots<{"__dict__" to #py.dict<{}>}>>
    %3 = py.constant #py.obj<type: @a, slots: #py.slots<{"__dict__" to #py.dict<{}>}>, value: #py.int<1>>
    %4 = py.constant #py.obj<type: @a, slots: #py.slots<{"__dict__" to #py.dict<{}>}>>
    return %4 : !py.dynamic
}
