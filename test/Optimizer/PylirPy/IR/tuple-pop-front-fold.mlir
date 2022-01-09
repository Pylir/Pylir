// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type

func @test() -> (!py.dynamic, !py.dynamic) {
    %0 = py.constant #py.tuple<(@builtins.tuple)>
    %element, %result = py.tuple.popFront %0
    return %element, %result : !py.dynamic, !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-DAG: %[[C1:.*]] = py.constant @builtins.tuple
// CHECK-DAG: %[[C2:.*]] = py.constant #py.tuple<()>
// CHECK: return %[[C1]], %[[C2]]
