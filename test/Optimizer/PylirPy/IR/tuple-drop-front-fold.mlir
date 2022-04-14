// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type

func @test() -> !py.dynamic {
    %0 = py.constant(#py.tuple<(@builtins.tuple)>)
    %1 = arith.constant 1 : index
    %result = py.tuple.dropFront %1, %0
    return %result : !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-DAG: %[[C2:.*]] = py.constant(#py.tuple<()>)
// CHECK: return %[[C2]]

func @test2(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeTuple (%arg0, %arg1)
    %1 = arith.constant 1 : index
    %result = py.tuple.dropFront %1, %0
    return %result : !py.dynamic
}

// CHECK-LABEL: @test2
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = py.makeTuple (%[[ARG1]])
// CHECK-NEXT: return %[[C]]
