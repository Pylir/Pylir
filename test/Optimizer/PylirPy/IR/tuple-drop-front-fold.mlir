// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type

py.func @test() -> !py.dynamic {
    %0 = constant(#py.tuple<(#py.ref<@builtins.tuple>)>)
    %1 = arith.constant 1 : index
    %result = tuple_dropFront %1, %0
    return %result : !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-DAG: %[[C2:.*]] = constant(#py.tuple<()>)
// CHECK: return %[[C2]]

py.func @test2(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = makeTuple (%arg0, %arg1)
    %1 = arith.constant 1 : index
    %result = tuple_dropFront %1, %0
    return %result : !py.dynamic
}

// CHECK-LABEL: @test2
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = makeTuple (%[[ARG1]])
// CHECK-NEXT: return %[[C]]

py.func @test3(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = arith.constant 0 : index
    %result = tuple_dropFront %0, %arg0
    return %result : !py.dynamic
}

// CHECK-LABEL: @test3
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = constant(#py.ref<@builtins.tuple>)
// CHECK-NEXT: %[[TUPLE:.*]] = tuple_copy %[[ARG0]] : %[[C]]
// CHECK-NEXT: return %[[TUPLE]]

py.func @test4(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = arith.constant 1 : index
    %1 = tuple_dropFront %0, %arg0
    %2 = tuple_dropFront %0, %1
    return %2 : !py.dynamic
}

// CHECK-LABEL: @test4
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = arith.constant 2 : index
// CHECK-NEXT: %[[TUPLE:.*]] = tuple_dropFront %[[C]], %[[ARG0]]
// CHECK-NEXT: return %[[TUPLE]]

py.func @test5(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = makeTuple (*%arg0)
    %1 = arith.constant 1 : index
    %result = tuple_dropFront %1, %0
    return %result : !py.dynamic
}

// CHECK-LABEL: @test5
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[TUPLE:.*]] = makeTuple (*%[[ARG0]])
// CHECK: %[[DROPPED:.*]] = tuple_dropFront %{{.*}}, %[[TUPLE]]
// CHECK: return %[[DROPPED]]

py.func @test6(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = tuple_prepend %arg0, %arg1
    %1 = arith.constant 1 : index
    %result = tuple_dropFront %1, %0
    return %result : !py.dynamic
}

// CHECK-LABEL: @test6
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK: %[[C:.*]] = constant(#py.ref<@builtins.tuple>)
// CHECK: %[[COPY:.*]] = tuple_copy %[[ARG1]] : %[[C]]
// CHECK: return %[[COPY]]
