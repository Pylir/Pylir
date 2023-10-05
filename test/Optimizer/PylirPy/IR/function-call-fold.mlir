// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.func @foo(%arg0 : !py.dynamic, %arg1 : !py.dynamic, %arg2 : !py.dynamic) -> !py.dynamic {
    return %arg0 : !py.dynamic
}

#test_function = #py.globalValue<test_function, initializer = #py.function<@foo>>

py.func @test(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = constant(#test_function)
    %1 = function_call %0(%0, %arg0, %arg1)
    return %1 : !py.dynamic
}

// CHECK: #[[$TEST_FUNCTION:.*]] = #py.globalValue<test_function{{.*}}>

// CHECK-LABEL: func @test(
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-DAG: %[[CLOSURE:.*]] = constant(#[[$TEST_FUNCTION]])
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
