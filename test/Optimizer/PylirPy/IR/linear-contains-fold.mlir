// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type

func @test1(%arg0 : !py.unknown) -> i1 {
    %0 = py.constant(@builtins.tuple) : !py.unknown
    %1 = py.makeTuple (* %arg0, %0) : (!py.unknown, !py.unknown) -> !py.unknown
    %2 = py.linearContains %0 in %1 : !py.unknown, !py.unknown
    return %2 : i1
}

// CHECK-LABEL: @test1
// CHECK-DAG: %[[C1:.*]] = arith.constant true
// CHECK: return %[[C1]]

func @test2(%arg0 : !py.unknown) -> i1 {
    %0 = py.constant(@builtins.tuple) : !py.unknown
    %1 = py.tuple.prepend %0, %arg0 : (!py.unknown, !py.unknown) -> !py.unknown
    %2 = py.linearContains %0 in %1 : !py.unknown, !py.unknown
    return %2 : i1
}

// CHECK-LABEL: @test2
// CHECK-DAG: %[[C1:.*]] = arith.constant true
// CHECK: return %[[C1]]
