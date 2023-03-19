// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue const @foo = #py.tuple<(#py.str<"__slots__">)>
py.globalValue const @builtins.type = #py.type<instance_slots = <(#py.str<"__slots__">)>, slots = { __slots__ = #py.ref<@foo> }>
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.str = #py.type

py.func @test1() -> !py.dynamic {
    %0 = constant(#py.tuple<(#py.ref<@builtins.type>)>)
    %c0 = arith.constant 0 : index
    %1 = mroLookup %c0 in %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: func @test1
// CHECK: %[[C1:.*]] = constant(#py.ref<@foo>)
// CHECK: return %[[C1]]

py.func @test2() -> !py.dynamic {
    %0 = constant(#py.tuple<()>)
    %c0 = arith.constant 0 : index
    %1 = mroLookup %c0 in %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: func @test2
// CHECK: %[[C1:.*]] = constant(#py.unbound)
// CHECK: return %[[C1]]

py.func @test3(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = constant(#py.ref<@builtins.type>)
    %1 = makeTuple (%0, %arg0)
    %c0 = arith.constant 0 : index
    %2 = mroLookup %c0 in %1
    return %2 : !py.dynamic
}

// CHECK-LABEL: func @test3
// CHECK-DAG: %[[C1:.*]] = constant(#py.ref<@foo>)
// CHECK: return %[[C1]]
