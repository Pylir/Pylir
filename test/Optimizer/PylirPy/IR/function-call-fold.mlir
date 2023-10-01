// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_function = #py.globalValue<builtins.function, initializer = #py.type>
py.external @builtins.function, #builtins_function
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_dict= #py.globalValue<builtins.dict, initializer = #py.type>
py.external @builtins.dict, #builtins_dict
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str
#builtins_None = #py.globalValue<builtins.None, initializer = #py.type>
py.external @builtins.None, #builtins_None

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
