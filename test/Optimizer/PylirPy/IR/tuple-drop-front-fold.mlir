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

func @test3(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = arith.constant 0 : index
    %result = py.tuple.dropFront %0, %arg0
    return %result : !py.dynamic
}

// CHECK-LABEL: @test3
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = py.constant(@builtins.tuple)
// CHECK-NEXT: %[[TUPLE:.*]] = py.tuple.copy %[[ARG0]] : %[[C]]
// CHECK-NEXT: return %[[TUPLE]]

func @test4(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = arith.constant 1 : index
    %1 = py.tuple.dropFront %0, %arg0
    %2 = py.tuple.dropFront %0, %1
    return %2 : !py.dynamic
}

// CHECK-LABEL: @test4
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = arith.constant 2 : index
// CHECK-NEXT: %[[TUPLE:.*]] = py.tuple.dropFront %[[C]], %[[ARG0]]
// CHECK-NEXT: return %[[TUPLE]]

func @test5(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeTuple (*%arg0)
    %1 = arith.constant 1 : index
    %result = py.tuple.dropFront %1, %0
    return %result : !py.dynamic
}

// CHECK-LABEL: @test5
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[TUPLE:.*]] = py.makeTuple (*%[[ARG0]])
// CHECK: %[[DROPPED:.*]] = py.tuple.dropFront %{{.*}}, %[[TUPLE]]
// CHECK: return %[[DROPPED]]

func @test6(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = py.tuple.prepend %arg0, %arg1
    %1 = arith.constant 1 : index
    %result = py.tuple.dropFront %1, %0
    return %result : !py.dynamic
}

// CHECK-LABEL: @test6
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK: %[[C:.*]] = py.constant(@builtins.tuple)
// CHECK: %[[COPY:.*]] = py.tuple.copy %[[ARG1]] : %[[C]]
// CHECK: return %[[COPY]]
