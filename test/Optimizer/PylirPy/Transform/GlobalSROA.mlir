// RUN: pylir-opt %s --pylir-global-sroa --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.dict = #py.type
py.globalValue @builtins.str = #py.type

py.globalValue "private" @thing = #py.dict<{}>

func.func @test(%hash: index) -> !py.dynamic {
    %0 = py.constant(#py.ref<@thing>)
    %1 = py.constant(#py.str<"lol">)
    %2 = py.dict.tryGetItem %0[%1 hash(%hash)]
    return %2 : !py.dynamic
}

func.func @foo(%hash: index, %arg0 : !py.dynamic) {
    %0 = py.constant(#py.ref<@thing>)
    %1 = py.constant(#py.str<"lol">)
    py.dict.setItem %0[%1 hash(%hash)] to %arg0
    return
}

// CHECK: py.global "private" @[[$DES:.*]] : !py.dynamic

// CHECK-LABEL: func.func @test
// CHECK: %[[LOAD:.*]] = py.load @[[$DES]] : !py.dynamic
// CHECK-NOT: py.dict.tryGetItem
// CHECK-NEXT: return %[[LOAD]]

// CHECK-LABEL: func.func @foo
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK: py.store %[[ARG1]] : !py.dynamic into @[[$DES]]
// CHECK-NOT: py.dict.setItem

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.dict = #py.type
py.globalValue @builtins.str = #py.type

py.globalValue "private" @thing = #py.dict<{}>

func.func @store_only(%hash: index, %arg0 : !py.dynamic) {
    %0 = py.constant(#py.ref<@thing>)
    %1 = py.constant(#py.str<"lol">)
    py.dict.setItem %0[%1 hash(%hash)] to %arg0
    return
}

// CHECK: py.global "private" @[[$DES:.*]] : !py.dynamic

// CHECK-LABEL: func.func @store_only
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK: py.store %[[ARG1]] : !py.dynamic into @[[$DES]]
// CHECK-NOT: py.dict.setItem

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.dict = #py.type
py.globalValue @builtins.str = #py.type

py.globalValue "private" @thing = #py.dict<{}>

func.func @load_only(%hash: index) -> !py.dynamic {
    %0 = py.constant(#py.ref<@thing>)
    %1 = py.constant(#py.str<"lol">)
    %2 = py.dict.tryGetItem %0[%1 hash(%hash)]
    return %2 : !py.dynamic
}

// CHECK: py.global "private" @[[$DES:.*]] : !py.dynamic

// CHECK-LABEL: func.func @load_only
// CHECK: %[[LOAD:.*]] = py.load @[[$DES]] : !py.dynamic
// CHECK-NOT: py.dict.tryGetItem
// CHECK-NEXT: return %[[LOAD]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.dict = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.int = #py.type

py.globalValue "private" @thing = #py.dict<{#py.str<"lol"> to #py.int<5>}>

func.func @init_attr(%hash: index) -> !py.dynamic {
    %0 = py.constant(#py.ref<@thing>)
    %1 = py.constant(#py.str<"lol">)
    %2 = py.dict.tryGetItem %0[%1 hash(%hash)]
    return %2 : !py.dynamic
}

// CHECK-NOT: py.globalValue "private" thing

// CHECK: py.global "private" @[[$DES:.*]] : !py.dynamic = #py.int<5>

// CHECK-LABEL: func.func @init_attr
// CHECK: %[[LOAD:.*]] = py.load @[[$DES]] : !py.dynamic
// CHECK-NOT: py.dict.tryGetItem
// CHECK-NEXT: return %[[LOAD]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.dict = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.int = #py.type

py.globalValue "private" @thing = #py.dict<{#py.str<"lol"> to #py.int<5>}>

// CHECK-NOT: py.globalValue "private" thing

// CHECK: py.global "private" @[[DES:.*]] : !py.dynamic = #py.int<5>


// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.dict = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.int = #py.type

py.globalValue "private" @thing = #py.dict<{}>

func.func @sub_attr(%hash: index) -> !py.dynamic {
    %0 = py.constant(#py.dict<{#py.int<5> to #py.ref<@thing>}>)
    %1 = py.constant(#py.str<"lol">)
    %2 = py.dict.tryGetItem %0[%1 hash(%hash)]
    return %2 : !py.dynamic
}

// CHECK: py.globalValue "private" @thing = #py.dict<{}>
// CHECK: py.constant(#py.dict<{#py.int<5> to #py.ref<@thing>}>)

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.dict = #py.type
py.globalValue @builtins.float = #py.type
py.globalValue @builtins.int = #py.type

py.globalValue "private" @thing = #py.dict<{#py.int<3> to #py.int<5>}>

func.func @init_attr(%hash: index) -> !py.dynamic {
    %0 = py.constant(#py.ref<@thing>)
    %1 = py.constant(#py.float<3.0>)
    %2 = py.dict.tryGetItem %0[%1 hash(%hash)]
    return %2 : !py.dynamic
}

// CHECK-NOT: py.globalValue "private" thing

// CHECK: py.global "private" @[[$DES:.*]] : !py.dynamic = #py.int<5>

// CHECK-LABEL: func.func @init_attr
// CHECK: %[[LOAD:.*]] = py.load @[[$DES]] : !py.dynamic
// CHECK-NOT: py.dict.tryGetItem
// CHECK-NEXT: return %[[LOAD]]
