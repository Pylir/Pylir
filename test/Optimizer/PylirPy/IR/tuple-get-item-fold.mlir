// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type

func @test() -> !py.dynamic {
    %0 = py.constant(#py.tuple<(@builtins.tuple)>)
    %1 = arith.constant 0 : index
    %2 = py.tuple.getItem %0[%1]
    return %2 : !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-DAG: %[[C1:.*]] = py.constant(@builtins.tuple)
// CHECK: return %[[C1]]

func @test2(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.constant(@builtins.tuple)
    %1 = py.makeTuple (%0, * %arg0)
    %2 = arith.constant 0 : index
    %3 = py.tuple.getItem %1[%2]
    return %3 : !py.dynamic
}

// CHECK-LABEL: @test2
// CHECK-DAG: %[[C1:.*]] = py.constant(@builtins.tuple)
// CHECK: return %[[C1]]

func @test3(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.constant(@builtins.tuple)
    %1 = py.tuple.prepend %0, %arg0
    %2 = arith.constant 0 : index
    %3 = py.tuple.getItem %1[%2]
    return %3 : !py.dynamic
}

// CHECK-LABEL: @test3
// CHECK-DAG: %[[C1:.*]] = py.constant(@builtins.tuple)
// CHECK: return %[[C1]]
