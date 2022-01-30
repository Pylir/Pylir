// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type

func @test1(%arg0 : !py.dynamic) -> i1 {
    %0 = py.constant @builtins.tuple
    %1 = py.makeTuple (* %arg0, %0)
    %2 = py.linearContains %0 in %1
    return %2 : i1
}

// CHECK-LABEL: @test1
// CHECK-DAG: %[[C1:.*]] = arith.constant true
// CHECK: return %[[C1]]

func @test2(%arg0 : !py.dynamic) -> i1 {
    %0 = py.constant @builtins.tuple
    %1 = py.tuple.prepend %0, %arg0
    %2 = py.linearContains %0 in %1
    return %2 : i1
}

// CHECK-LABEL: @test2
// CHECK-DAG: %[[C1:.*]] = arith.constant true
// CHECK: return %[[C1]]
