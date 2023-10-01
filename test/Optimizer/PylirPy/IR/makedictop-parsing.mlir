// RUN: pylir-opt %s | pylir-opt | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int
#builtins_dict= #py.globalValue<builtins.dict, initializer = #py.type>
py.external @builtins.dict, #builtins_dict
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_list= #py.globalValue<builtins.list, initializer = #py.type>
py.external @builtins.list, #builtins_list

// CHECK-LABEL: @makedictop_test
py.func @makedictop_test(%hash: index) -> !py.dynamic {
    %0 = constant(#py.int<0>)
    %1 = constant(#py.dict<{#py.str<"a"> to #py.int<3>, #py.str<"b"> to #py.list<[#py.int<5>]>}>)
    %2 = constant(#py.str<"string">)
    %3 = constant(#py.tuple<(#py.int<0>,#py.int<2>)>)
    %4 = makeDict (%0 hash(%hash) : %2,**%1,%2 hash(%hash) : %3)
    return %4 : !py.dynamic
}

