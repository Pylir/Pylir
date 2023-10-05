// RUN: pylir-opt %s --pylir-global-sroa --split-input-file | FileCheck %s

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
