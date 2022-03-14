// RUN: pylir-opt %s | pylir-opt | FileCheck %s
// RUN: pylir-opt %s --mlir-print-op-generic | pylir-opt | FileCheck %s

// Stubs
py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.bool = #py.type<>
py.globalValue @builtins.int = #py.type<>
py.globalValue @builtins.float = #py.type<>
py.globalValue @builtins.str = #py.type<>
py.globalValue @builtins.list = #py.type<>
py.globalValue @builtins.tuple = #py.type<>
py.globalValue @builtins.set = #py.type<>
py.globalValue @builtins.dict = #py.type<>
py.globalValue @builtins.function = #py.type<>
py.globalValue @builtins.None = #py.type<>

func @foo(%arg0 : !py.dynamic, %arg1 : !py.dynamic, %arg2 : !py.dynamic) -> !py.dynamic {
    return %arg0 : !py.dynamic
}

py.globalValue @test_function = #py.function<value = @foo, kwDefaults = #py.dict<value = {}>>

// CHECK-LABEL: test_constant_integer
func @test_constant_integer() -> !py.dynamic {
    %0 = py.constant #py.int<value = 50>
    return %0 : !py.dynamic
}

// CHECK-LABEL: test_constant_bool
func @test_constant_bool() -> !py.dynamic {
    %0 = py.constant #py.bool<value = True>
    %1 = py.constant #py.bool<value = False>
    return %0 : !py.dynamic
}

// CHECK-LABEL: test_constant_float
func @test_constant_float() -> !py.dynamic {
    %0 = py.constant #py.float<value = 433.4>
    return %0 : !py.dynamic
}

// CHECK-LABEL: test_constant_string
func @test_constant_string() -> !py.dynamic {
    %0 = py.constant #py.str<value = "text">
    return %0 : !py.dynamic
}

// CHECK-LABEL: test_constant_list
func @test_constant_list() -> !py.dynamic {
    %0 = py.constant #py.list<value = [#py.float<value = 433.4>, #py.int<value = 5>]>
    %empty = py.constant #py.list<value = []>
    return %0 : !py.dynamic
}

// CHECK-LABEL: test_constant_tuple
func @test_constant_tuple() -> !py.dynamic {
    %0 = py.constant #py.tuple<value = (#py.float<value = 433.4>, #py.int<value = 5>)>
    %empty = py.constant #py.tuple<value = ()>
    return %0 : !py.dynamic
}

// CHECK-LABEL: test_constant_set
func @test_constant_set() -> !py.dynamic {
    %0 = py.constant #py.set<value = {#py.float<value = 433.4>, #py.int<value = 5>}>
    %empty = py.constant #py.set<value = {}>
    return %0 : !py.dynamic
}

// CHECK-LABEL: test_constant_dict
func @test_constant_dict() -> !py.dynamic {
    %0 = py.constant #py.dict<value = {#py.float<value = 433.4> to #py.int<value = 5>, #py.str<value = "__call__"> to #py.int<value = 5>}>
    %empty = py.constant #py.dict<value = {}>
    return %0 : !py.dynamic
}

py.globalValue @a = #py.type<>

// CHECK-LABEL: test_objects
func @test_objects() -> !py.dynamic {
    %0 = py.constant #py.obj<typeObject = @a>
    %1 = py.constant #py.obj<typeObject = @a, builtinValue = #py.int<value = 1>>
    %2 = py.constant #py.obj<typeObject = @a, builtinValue = #py.int<value = 1>, slots = {__dict__ = #py.dict<value = {}>}>
    %3 = py.constant #py.obj<typeObject = @a, slots = {__dict__ = #py.dict<value = {}>}, builtinValue = #py.int<value = 1>>
    %4 = py.constant #py.obj<typeObject = @a, slots = {__dict__ = #py.dict<value = {}>}>
    return %4 : !py.dynamic
}
