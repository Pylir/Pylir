// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

// Stubs
py.globalValue @builtins.type = #py.type
py.globalValue @builtins.function = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.dict = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.None = #py.type

func.func @foo(%arg0 : !py.dynamic, %arg1 : !py.dynamic, %arg2 : !py.dynamic) -> !py.dynamic {
    return %arg0 : !py.dynamic
}

py.globalValue @test_function = #py.function<@foo>

func.func @test(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = py.constant(@test_function)
    %1 = py.function.call %0(%0, %arg0, %arg1)
    return %1 : !py.dynamic
}

// CHECK-LABEL: func @test(
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-DAG: %[[CLOSURE:.*]] = py.constant(@test_function)
// CHECK-NEXT: %[[RESULT:.*]] = py.call @foo(%[[CLOSURE]], %[[ARG0]], %[[ARG1]])
// CHECK-NEXT: return %[[RESULT]]

func.func @test2(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeFunc @foo
    %1 = py.function.call %0(%0, %arg0, %arg1)
    return %1 : !py.dynamic
}

// CHECK-LABEL: func @test2(
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-DAG: %[[CLOSURE:.*]] = py.makeFunc @foo
// CHECK-NEXT: %[[RESULT:.*]] = py.call @foo(%[[CLOSURE]], %[[ARG0]], %[[ARG1]])
// CHECK-NEXT: return %[[RESULT]]
