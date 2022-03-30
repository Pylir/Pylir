// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type

func @test() -> !py.unknown {
    %0 = py.constant(#py.tuple<value = (@builtins.tuple)>) : !py.unknown
    %1 = arith.constant 0 : index
    %2 = py.tuple.getItem %0[%1] : (!py.unknown) -> !py.unknown
    return %2 : !py.unknown
}

// CHECK-LABEL: @test
// CHECK-DAG: %[[C1:.*]] = py.constant(@builtins.tuple)
// CHECK: return %[[C1]]

func @test2(%arg0 : !py.unknown) -> !py.unknown {
    %0 = py.constant(@builtins.tuple) : !py.unknown
    %1 = py.makeTuple (%0, * %arg0) : (!py.unknown, !py.unknown) -> !py.unknown
    %2 = arith.constant 0 : index
    %3 = py.tuple.getItem %1[%2] : (!py.unknown) -> !py.unknown
    return %3 : !py.unknown
}

// CHECK-LABEL: @test2
// CHECK-DAG: %[[C1:.*]] = py.constant(@builtins.tuple) : !py.unknown
// CHECK: return %[[C1]]

func @test3(%arg0 : !py.unknown) -> !py.unknown {
    %0 = py.constant(@builtins.tuple) : !py.unknown
    %1 = py.tuple.prepend %0, %arg0 : (!py.unknown, !py.unknown) -> !py.unknown
    %2 = arith.constant 0 : index
    %3 = py.tuple.getItem %1[%2] : (!py.unknown) -> !py.unknown
    return %3 : !py.unknown
}

// CHECK-LABEL: @test3
// CHECK-DAG: %[[C1:.*]] = py.constant(@builtins.tuple)
// CHECK: return %[[C1]]
