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

py.globalValue @test_function = #py.function<@foo, kwDefaults = #py.dict<{}>>

// CHECK-LABEL: test_constant_integer
func @test_constant_integer() -> !py.unknown {
    %0 = py.constant(#py.int<50>) : !py.unknown
    return %0 : !py.unknown
}

// CHECK-LABEL: test_constant_bool
func @test_constant_bool() -> !py.unknown {
    %0 = py.constant(#py.bool<True>) : !py.unknown
    %1 = py.constant(#py.bool<False>) : !py.unknown
    return %0 : !py.unknown
}

// CHECK-LABEL: test_constant_float
func @test_constant_float() -> !py.unknown {
    %0 = py.constant(#py.float<433.4>) : !py.unknown
    return %0 : !py.unknown
}

// CHECK-LABEL: test_constant_string
func @test_constant_string() -> !py.unknown {
    %0 = py.constant(#py.str<"text">) : !py.unknown
    return %0 : !py.unknown
}

// CHECK-LABEL: test_constant_list
func @test_constant_list() -> !py.unknown {
    %0 = py.constant(#py.list<[#py.float<433.4>, #py.int<5>]>) : !py.unknown
    %empty = py.constant(#py.list<[]>) : !py.unknown
    return %0 : !py.unknown
}

// CHECK-LABEL: test_constant_tuple
func @test_constant_tuple() -> !py.unknown {
    %0 = py.constant(#py.tuple<(#py.float<433.4>, #py.int<5>)>) : !py.unknown
    %empty = py.constant(#py.tuple<()>) : !py.unknown
    return %0 : !py.unknown
}

// CHECK-LABEL: test_constant_set
func @test_constant_set() -> !py.unknown {
    %0 = py.constant(#py.set<{#py.float<433.4>, #py.int<5>}>) : !py.unknown
    %empty = py.constant(#py.set<{}>) : !py.unknown
    return %0 : !py.unknown
}

// CHECK-LABEL: test_constant_dict
func @test_constant_dict() -> !py.unknown {
    %0 = py.constant(#py.dict<{#py.float<433.4> to #py.int<5>, #py.str<"__call__"> to #py.int<5>}>) : !py.unknown
    %empty = py.constant(#py.dict<{}>) : !py.unknown
    return %0 : !py.unknown
}

py.globalValue @a = #py.type

// CHECK-LABEL: test_objects
func @test_objects() -> !py.unknown {
    %0 = py.constant(#py.obj<@a>) : !py.unknown
    %1 = py.constant(#py.obj<@a, {__dict__ = #py.dict<{}>}>) : !py.unknown
    return %1 : !py.unknown
}
