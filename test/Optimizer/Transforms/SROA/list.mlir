// RUN: pylir-opt -pass-pipeline="builtin.module(any(pylir-sroa))" %s --split-input-file | FileCheck %s

py.func @test(%arg0 : !py.dynamic) -> (!py.dynamic, index) {
    %0 = constant(#py.str<"Hello">)
    %1 = constant(#py.str<" ">)
    %2 = constant(#py.str<"World">)
    %l = makeList (%0, %1, %2, %arg0)
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %three = arith.constant 3 : index
    %3 = list_getItem %l[%zero]
    %4 = list_getItem %l[%one]
    %5 = list_getItem %l[%two]
    %6 = list_getItem %l[%three]
    %7 = str_concat %3, %4, %5, %6
    %8 = list_len %l
    return %7, %8 : !py.dynamic, index
}

// CHECK-LABEL: py.func @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-DAG: %[[H:.*]] = constant(#py.str<"Hello">)
// CHECK-DAG: %[[S:.*]] = constant(#py.str<" ">)
// CHECK-DAG: %[[W:.*]] = constant(#py.str<"World">)
// CHECK-DAG: %[[L:.*]] = arith.constant 4
// CHECK: %[[R:.*]] = str_concat %[[H]], %[[S]], %[[W]], %[[ARG0]]
// CHECK: return %[[R]], %[[L]]

// -----

py.func @test(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = constant(#py.str<"Hello">)
    %zero = arith.constant 0 : index
    %l = makeList (%arg0)
    %1 = test.random
    cf.cond_br %1, ^bb0, ^bb1

^bb0:
    list_setItem %l[%zero] to %0
    cf.br ^bb1

^bb1:
    %2 = list_getItem %l[%zero]
    return %2 : !py.dynamic
}

// CHECK-LABEL: py.func @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[H:.*]] = constant(#py.str<"Hello">)
// CHECK: cf.cond_br %{{.*}}, ^[[COND:.*]], ^[[RET:.*]](%[[ARG0]] : !py.dynamic)

// CHECK: ^[[COND]]:
// CHECK-NEXT: cf.br ^[[RET]](%[[H]] : !py.dynamic)

// CHECK: ^[[RET]]
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: return %[[ARG]]

// -----

py.func @neg_test(%arg0 : !py.dynamic, %arg1 : index) -> (!py.dynamic, index) {
    %0 = constant(#py.str<"Hello">)
    %1 = constant(#py.str<" ">)
    %2 = constant(#py.str<"World">)
    %l = makeList (%0, %1, %2, %arg0)
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %3 = list_getItem %l[%zero]
    %4 = list_getItem %l[%one]
    %5 = list_getItem %l[%two]
    %6 = list_getItem %l[%arg1]
    %7 = str_concat %3, %4, %5, %6
    %8 = list_len %l
    return %7, %8 : !py.dynamic, index
}

// CHECK-LABEL: py.func @neg_test
// CHECK: %[[L:.*]] = makeList
// CHECK: %[[ZERO:.*]] = list_getItem %[[L]]
// CHECK: %[[ONE:.*]] = list_getItem %[[L]]
// CHECK: %[[TWO:.*]] = list_getItem %[[L]]
// CHECK: %[[THREE:.*]] = list_getItem %[[L]]
// CHECK: %[[S:.*]] = str_concat %[[ZERO]], %[[ONE]], %[[TWO]], %[[THREE]]
// CHECK: %[[LEN:.*]] = list_len %[[L]]
// CHECK: return %[[S]], %[[LEN]]
