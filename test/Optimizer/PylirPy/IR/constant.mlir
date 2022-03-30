// RUN: pylir-opt %s | pylir-opt | FileCheck %s
// RUN: pylir-opt %s --mlir-print-op-generic | pylir-opt | FileCheck %s

// Stubs
py.globalValue @builtins.type = #py.type
py.globalValue @builtins.bool = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.float = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.list = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.set = #py.type
py.globalValue @builtins.dict = #py.type
py.globalValue @builtins.function = #py.type
py.globalValue @builtins.None = #py.type

func @foo(%arg0 : !py.unknown, %arg1 : !py.unknown, %arg2 : !py.unknown) -> !py.unknown {
    return %arg0 : !py.unknown
}

py.globalValue @test_function = #py.function<value = @foo, kwDefaults = #py.dict<value = {}>>

// CHECK-LABEL: test_constant_integer
func @test_constant_integer() -> !py.unknown {
    %0 = py.constant(#py.int<value = 50>) : !py.unknown
    return %0 : !py.unknown
}

// CHECK-LABEL: test_constant_bool
func @test_constant_bool() -> !py.unknown {
    %0 = py.constant(#py.bool<value = True>) : !py.unknown
    %1 = py.constant(#py.bool<value = False>) : !py.unknown
    return %0 : !py.unknown
}

// CHECK-LABEL: test_constant_float
func @test_constant_float() -> !py.unknown {
    %0 = py.constant(#py.float<value = 433.4>) : !py.unknown
    return %0 : !py.unknown
}

// CHECK-LABEL: test_constant_string
func @test_constant_string() -> !py.unknown {
    %0 = py.constant(#py.str<value = "text">) : !py.unknown
    return %0 : !py.unknown
}

// CHECK-LABEL: test_constant_list
func @test_constant_list() -> !py.unknown {
    %0 = py.constant(#py.list<value = [#py.float<value = 433.4>, #py.int<value = 5>]>) : !py.unknown
    %empty = py.constant(#py.list<value = []>) : !py.unknown
    return %0 : !py.unknown
}

// CHECK-LABEL: test_constant_tuple
func @test_constant_tuple() -> !py.unknown {
    %0 = py.constant(#py.tuple<value = (#py.float<value = 433.4>, #py.int<value = 5>)>) : !py.unknown
    %empty = py.constant(#py.tuple<value = ()>) : !py.unknown
    return %0 : !py.unknown
}

// CHECK-LABEL: test_constant_set
func @test_constant_set() -> !py.unknown {
    %0 = py.constant(#py.set<value = {#py.float<value = 433.4>, #py.int<value = 5>}>) : !py.unknown
    %empty = py.constant(#py.set<value = {}>) : !py.unknown
    return %0 : !py.unknown
}

// CHECK-LABEL: test_constant_dict
func @test_constant_dict() -> !py.unknown {
    %0 = py.constant(#py.dict<value = {#py.float<value = 433.4> to #py.int<value = 5>, #py.str<value = "__call__"> to #py.int<value = 5>}>) : !py.unknown
    %empty = py.constant(#py.dict<value = {}>) : !py.unknown
    return %0 : !py.unknown
}

py.globalValue @a = #py.type<>

// CHECK-LABEL: test_objects
func @test_objects() -> !py.unknown {
    %0 = py.constant(#py.obj<typeObject = @a>) : !py.unknown
    %1 = py.constant(#py.obj<typeObject = @a, builtinValue = #py.int<value = 1>>) : !py.unknown
    %2 = py.constant(#py.obj<typeObject = @a, builtinValue = #py.int<value = 1>, slots = {__dict__ = #py.dict<value = {}>}>) : !py.unknown
    %3 = py.constant(#py.obj<typeObject = @a, slots = {__dict__ = #py.dict<value = {}>}, builtinValue = #py.int<value = 1>>) : !py.unknown
    %4 = py.constant(#py.obj<typeObject = @a, slots = {__dict__ = #py.dict<value = {}>}>) : !py.unknown
    return %4 : !py.unknown
}
