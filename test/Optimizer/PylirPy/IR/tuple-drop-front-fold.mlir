// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type

func @test() -> !py.unknown {
    %0 = py.constant(#py.tuple<(@builtins.tuple)>) : !py.unknown
    %1 = arith.constant 1 : index
    %result = py.tuple.dropFront %1, %0 : (!py.unknown) -> !py.unknown
    return %result : !py.unknown
}

// CHECK-LABEL: @test
// CHECK-DAG: %[[C2:.*]] = py.constant(#py.tuple<()>)
// CHECK: return %[[C2]]

func @test2(%arg0 : !py.unknown, %arg1 : !py.unknown) -> !py.unknown {
    %0 = py.makeTuple (%arg0, %arg1) : (!py.unknown, !py.unknown) -> !py.unknown
    %1 = arith.constant 1 : index
    %result = py.tuple.dropFront %1, %0 : (!py.unknown) -> !py.unknown
    return %result : !py.unknown
}

// CHECK-LABEL: @test2
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = py.makeTuple (%[[ARG1]])
// CHECK-NEXT: return %[[C]]
