// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>

py.func @test1(%arg0 : !py.dynamic) -> i1 {
    %0 = constant(#builtins_tuple)
    %1 = makeTuple (* %arg0, %0)
    %2 = tuple_contains %0 in %1
    return %2 : i1
}

// CHECK-LABEL: @test1
// CHECK-DAG: %[[C1:.*]] = arith.constant true
// CHECK: return %[[C1]]

py.func @test2(%arg0 : !py.dynamic) -> i1 {
    %0 = constant(#builtins_tuple)
    %1 = tuple_prepend %0, %arg0
    %2 = tuple_contains %0 in %1
    return %2 : i1
}

// CHECK-LABEL: @test2
// CHECK-DAG: %[[C1:.*]] = arith.constant true
// CHECK: return %[[C1]]
