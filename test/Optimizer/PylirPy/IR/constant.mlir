// RUN: pylir-opt %s | pylir-opt | FileCheck %s
// RUN: pylir-opt %s --mlir-print-op-generic | pylir-opt | FileCheck %s

// Stubs
#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int
#builtins_float = #py.globalValue<builtins.float, initializer = #py.type>
py.external @builtins.float, #builtins_float
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str
#builtins_list= #py.globalValue<builtins.list, initializer = #py.type>
py.external @builtins.list, #builtins_list
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_dict= #py.globalValue<builtins.dict, initializer = #py.type>
py.external @builtins.dict, #builtins_dict
#builtins_function = #py.globalValue<builtins.function, initializer = #py.type>
py.external @builtins.function, #builtins_function
#builtins_None = #py.globalValue<builtins.None, initializer = #py.type>
py.external @builtins.None, #builtins_None

py.func @foo(%arg0 : !py.dynamic, %arg1 : !py.dynamic, %arg2 : !py.dynamic) -> !py.dynamic {
    return %arg0 : !py.dynamic
}

// CHECK-LABEL: test_function
py.func @test_function() -> !py.dynamic {
  %0 = constant(#py.function<@foo, kw_defaults = #py.dict<{}>>)
  return %0 : !py.dynamic
}

// CHECK-LABEL: test_constant_integer
py.func @test_constant_integer() -> !py.dynamic {
    %0 = constant(#py.int<50>)
    return %0 : !py.dynamic
}

// CHECK-LABEL: test_constant_bool
py.func @test_constant_bool() -> !py.dynamic {
    %0 = constant(#py.bool<True>)
    %1 = constant(#py.bool<False>)
    return %0 : !py.dynamic
}

// CHECK-LABEL: test_constant_float
py.func @test_constant_float() -> !py.dynamic {
    %0 = constant(#py.float<433.4>)
    return %0 : !py.dynamic
}

// CHECK-LABEL: test_constant_string
py.func @test_constant_string() -> !py.dynamic {
    %0 = constant(#py.str<"text">)
    return %0 : !py.dynamic
}

// CHECK-LABEL: test_constant_list
py.func @test_constant_list() -> !py.dynamic {
    %0 = constant(#py.list<[#py.float<433.4>, #py.int<5>]>)
    %empty = constant(#py.list<[]>)
    return %0 : !py.dynamic
}

// CHECK-LABEL: test_constant_tuple
py.func @test_constant_tuple() -> !py.dynamic {
    %0 = constant(#py.tuple<(#py.float<433.4>, #py.int<5>)>)
    %empty = constant(#py.tuple<()>)
    return %0 : !py.dynamic
}

// CHECK-LABEL: test_constant_dict
py.func @test_constant_dict() -> !py.dynamic {
    %0 = constant(#py.dict<{#py.float<433.4> to #py.int<5>, #py.str<"__call__"> to #py.int<5>}>)
    %empty = constant(#py.dict<{}>)
    return %0 : !py.dynamic
}

#a = #py.globalValue<a, initializer = #py.type>

// CHECK-LABEL: test_objects
py.func @test_objects() -> !py.dynamic {
    %0 = constant(#py.obj<#a>)
    %1 = constant(#py.obj<#a, {__dict__ = #py.dict<{}>}>)
    return %1 : !py.dynamic
}
