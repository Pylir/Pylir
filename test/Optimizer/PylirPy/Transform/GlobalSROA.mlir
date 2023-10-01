// RUN: pylir-opt %s --pylir-global-sroa --split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_dict= #py.globalValue<builtins.dict, initializer = #py.type>
py.external @builtins.dict, #builtins_dict
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str

#thing = #py.globalValue<thing, initializer = #py.dict<{}>>

py.func @test(%hash: index) -> !py.dynamic {
    %0 = constant(#thing)
    %1 = constant(#py.str<"lol">)
    %2 = dict_tryGetItem %0[%1 hash(%hash)]
    return %2 : !py.dynamic
}

py.func @foo(%hash: index, %arg0 : !py.dynamic) {
    %0 = constant(#thing)
    %1 = constant(#py.str<"lol">)
    dict_setItem %0[%1 hash(%hash)] to %arg0
    return
}

// CHECK-LABEL: py.func @test
// CHECK: %[[LOAD:.*]] = load @[[$DES:.*]] : !py.dynamic
// CHECK-NOT: dict_tryGetItem
// CHECK-NEXT: return %[[LOAD]]

// CHECK-LABEL: py.func @foo
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK: store %[[ARG1]] : !py.dynamic into @[[$DES]]
// CHECK-NOT: dict_setItem

// CHECK: global "private" @[[$DES]] : !py.dynamic

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_dict= #py.globalValue<builtins.dict, initializer = #py.type>
py.external @builtins.dict, #builtins_dict
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str

#thing = #py.globalValue<thing, initializer = #py.dict<{}>>

py.func @store_only(%hash: index, %arg0 : !py.dynamic) {
    %0 = constant(#thing)
    %1 = constant(#py.str<"lol">)
    dict_setItem %0[%1 hash(%hash)] to %arg0
    return
}

// CHECK-LABEL: py.func @store_only
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK: store %[[ARG1]] : !py.dynamic into @[[$DES:.*]]
// CHECK-NOT: dict_setItem

// CHECK: global "private" @[[$DES]] : !py.dynamic

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_dict= #py.globalValue<builtins.dict, initializer = #py.type>
py.external @builtins.dict, #builtins_dict
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str

#thing = #py.globalValue<thing, initializer = #py.dict<{}>>

py.func @load_only(%hash: index) -> !py.dynamic {
    %0 = constant(#thing)
    %1 = constant(#py.str<"lol">)
    %2 = dict_tryGetItem %0[%1 hash(%hash)]
    return %2 : !py.dynamic
}

// CHECK-LABEL: py.func @load_only
// CHECK: %[[LOAD:.*]] = load @[[$DES:.*]] : !py.dynamic
// CHECK-NOT: dict_tryGetItem
// CHECK-NEXT: return %[[LOAD]]

// CHECK: global "private" @[[$DES]] : !py.dynamic

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_dict= #py.globalValue<builtins.dict, initializer = #py.type>
py.external @builtins.dict, #builtins_dict
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int

#thing = #py.globalValue<thing, initializer = #py.dict<{#py.str<"lol"> to #py.int<5>}>>

py.func @init_attr(%hash: index) -> !py.dynamic {
    %0 = constant(#thing)
    %1 = constant(#py.str<"lol">)
    %2 = dict_tryGetItem %0[%1 hash(%hash)]
    return %2 : !py.dynamic
}

// CHECK-NOT: globalValue<thing

// CHECK-LABEL: py.func @init_attr
// CHECK: %[[LOAD:.*]] = load @[[$DES:.*]] : !py.dynamic
// CHECK-NOT: dict_tryGetItem
// CHECK-NEXT: return %[[LOAD]]

// CHECK: global "private" @[[$DES]] : !py.dynamic = #py.int<5>

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_dict= #py.globalValue<builtins.dict, initializer = #py.type>
py.external @builtins.dict, #builtins_dict
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int

#thing = #py.globalValue<thing, initializer = #py.dict<{}>>

py.func @sub_attr(%hash: index) -> !py.dynamic {
    %0 = constant(#py.dict<{#py.int<5> to #thing}>)
    %1 = constant(#py.str<"lol">)
    %2 = dict_tryGetItem %0[%1 hash(%hash)]
    return %2 : !py.dynamic
}

// CHECK: globalValue<thing, initializer = #py.dict<{}>>
// CHECK: constant(#py.dict<{#py.int<5> to #thing}>)

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_dict= #py.globalValue<builtins.dict, initializer = #py.type>
py.external @builtins.dict, #builtins_dict
#builtins_float = #py.globalValue<builtins.float, initializer = #py.type>
py.external @builtins.float, #builtins_float
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int

#thing = #py.globalValue<thing, initializer = #py.dict<{#py.int<3> to #py.int<5>}>>

py.func @init_attr(%hash: index) -> !py.dynamic {
    %0 = constant(#thing)
    %1 = constant(#py.float<3.0>)
    %2 = dict_tryGetItem %0[%1 hash(%hash)]
    return %2 : !py.dynamic
}

// CHECK-NOT: globalValue<thing

// CHECK-LABEL: py.func @init_attr
// CHECK: %[[LOAD:.*]] = load @[[$DES:.*]] : !py.dynamic
// CHECK-NOT: dict_tryGetItem
// CHECK-NEXT: return %[[LOAD]]

// CHECK: global "private" @[[$DES]] : !py.dynamic = #py.int<5>
