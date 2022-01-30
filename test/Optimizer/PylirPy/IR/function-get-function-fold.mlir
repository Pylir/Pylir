// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

func private @foo(!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.None = #py.type
py.globalValue @builtins.function = #py.type
py.globalValue @builtins.str = #py.type

py.globalValue @bar = #py.function<@foo>

func @test() -> ((!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic) {
    %0 = py.constant @bar
    %1 = py.function.getFunction %0
    return %1 : (!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic
}

// CHECK-LABEL: @test
// CHECK: %[[RESULT:.*]] = constant @foo
// CHECK: return %[[RESULT]]

func @test2() -> ((!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic) {
    %0 = py.makeFunc @foo
    %1 = py.function.getFunction %0
    return %1 : (!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic
}

// CHECK-LABEL: @test2
// CHECK: %[[RESULT:.*]] = constant @foo
// CHECK: return %[[RESULT]]
