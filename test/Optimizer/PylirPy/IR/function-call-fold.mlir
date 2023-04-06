// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

// Stubs
py.globalValue @builtins.type = #py.type
py.globalValue @builtins.function = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.dict = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.None = #py.type

py.func @foo(%arg0 : !py.dynamic, %arg1 : !py.dynamic, %arg2 : !py.dynamic) -> !py.dynamic {
    return %arg0 : !py.dynamic
}

py.globalValue @test_function = #py.function<@foo>

py.func @test(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = constant(#py.ref<@test_function>)
    %1 = function_call %0(%0, %arg0, %arg1)
    return %1 : !py.dynamic
}

// CHECK-LABEL: func @test(
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-DAG: %[[CLOSURE:.*]] = constant(#py.ref<@test_function>)
// CHECK-NEXT: %[[RESULT:.*]] = call @foo(%[[CLOSURE]], %[[ARG0]], %[[ARG1]])
// CHECK-NEXT: return %[[RESULT]]

py.func @test2(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = makeFunc @foo
    %1 = function_call %0(%0, %arg0, %arg1)
    return %1 : !py.dynamic
}

// CHECK-LABEL: func @test2(
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-DAG: %[[CLOSURE:.*]] = makeFunc @foo
// CHECK-NEXT: %[[RESULT:.*]] = call @foo(%[[CLOSURE]], %[[ARG0]], %[[ARG1]])
// CHECK-NEXT: return %[[RESULT]]
