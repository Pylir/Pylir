// RUN: pylir-opt -pass-pipeline="builtin.module(any(pylir-sroa))" %s --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type

func.func @test(%arg0 : !py.dynamic) -> (!py.dynamic, index) {
    %0 = py.constant(#py.str<"Hello">)
    %1 = py.constant(#py.str<" ">)
    %2 = py.constant(#py.str<"World">)
    %l = py.makeList (%0, %1, %2, %arg0)
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %three = arith.constant 3 : index
    %3 = py.list.getItem %l[%zero]
    %4 = py.list.getItem %l[%one]
    %5 = py.list.getItem %l[%two]
    %6 = py.list.getItem %l[%three]
    %7 = py.str.concat %3, %4, %5, %6
    %8 = py.list.len %l
    return %7, %8 : !py.dynamic, index
}

// CHECK-LABEL: func.func @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-DAG: %[[H:.*]] = py.constant(#py.str<"Hello">)
// CHECK-DAG: %[[S:.*]] = py.constant(#py.str<" ">)
// CHECK-DAG: %[[W:.*]] = py.constant(#py.str<"World">)
// CHECK-DAG: %[[L:.*]] = arith.constant 4
// CHECK: %[[R:.*]] = py.str.concat %[[H]], %[[S]], %[[W]], %[[ARG0]]
// CHECK: return %[[R]], %[[L]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type

func.func @test(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.constant(#py.str<"Hello">)
    %zero = arith.constant 0 : index
    %l = py.makeList (%arg0)
    %1 = test.random
    cf.cond_br %1, ^bb0, ^bb1

^bb0:
    py.list.setItem %l[%zero] to %0
    cf.br ^bb1

^bb1:
    %2 = py.list.getItem %l[%zero]
    return %2 : !py.dynamic
}

// CHECK-LABEL: func.func @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[H:.*]] = py.constant(#py.str<"Hello">)
// CHECK: cf.cond_br %{{.*}}, ^[[COND:.*]], ^[[RET:.*]](%[[ARG0]] : !py.dynamic)

// CHECK: ^[[COND]]:
// CHECK-NEXT: cf.br ^[[RET]](%[[H]] : !py.dynamic)

// CHECK: ^[[RET]]
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: return %[[ARG]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type

func.func @neg_test(%arg0 : !py.dynamic, %arg1 : index) -> (!py.dynamic, index) {
    %0 = py.constant(#py.str<"Hello">)
    %1 = py.constant(#py.str<" ">)
    %2 = py.constant(#py.str<"World">)
    %l = py.makeList (%0, %1, %2, %arg0)
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %3 = py.list.getItem %l[%zero]
    %4 = py.list.getItem %l[%one]
    %5 = py.list.getItem %l[%two]
    %6 = py.list.getItem %l[%arg1]
    %7 = py.str.concat %3, %4, %5, %6
    %8 = py.list.len %l
    return %7, %8 : !py.dynamic, index
}

// CHECK-LABEL: func.func @neg_test
// CHECK: %[[L:.*]] = py.makeList
// CHECK: %[[ZERO:.*]] = py.list.getItem %[[L]]
// CHECK: %[[ONE:.*]] = py.list.getItem %[[L]]
// CHECK: %[[TWO:.*]] = py.list.getItem %[[L]]
// CHECK: %[[THREE:.*]] = py.list.getItem %[[L]]
// CHECK: %[[S:.*]] = py.str.concat %[[ZERO]], %[[ONE]], %[[TWO]], %[[THREE]]
// CHECK: %[[LEN:.*]] = py.list.len %[[L]]
// CHECK: return %[[S]], %[[LEN]]
