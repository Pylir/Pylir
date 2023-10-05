// RUN: pylir-opt -pass-pipeline="builtin.module(any(pylir-sroa))" %s --split-input-file | FileCheck %s

py.func @test(%arg0 : !py.dynamic, %hash : index) -> !py.dynamic {
    %0 = constant(#py.str<"Hello">)
    %1 = constant(#py.str<" ">)
    %2 = constant(#py.str<"World">)
    %zero = constant(#py.int<0>)
    %one = constant(#py.int<1>)
    %two = constant(#py.int<2>)
    %three = constant(#py.int<3>)
    %l = makeDict (%zero hash(%hash) : %0, %one hash(%hash) : %1, %two hash(%hash) : %2, %three hash(%hash) : %arg0)
    %3 = dict_tryGetItem %l[%zero hash(%hash)]
    %4 = dict_tryGetItem %l[%one hash(%hash)]
    %5 = dict_tryGetItem %l[%two hash(%hash)]
    %6 = dict_tryGetItem %l[%three hash(%hash)]
    %7 = str_concat %3, %4, %5, %6
    return %7 : !py.dynamic
}

// CHECK-LABEL: py.func @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-DAG: %[[H:.*]] = constant(#py.str<"Hello">)
// CHECK-DAG: %[[S:.*]] = constant(#py.str<" ">)
// CHECK-DAG: %[[W:.*]] = constant(#py.str<"World">)
// CHECK: %[[R:.*]] = str_concat %[[H]], %[[S]], %[[W]], %[[ARG0]]
// CHECK: return %[[R]]

// -----

py.func @test(%arg0 : !py.dynamic, %hash : index) -> (!py.dynamic, index) {
    %0 = constant(#py.str<"Hello">)
    %zero = constant(#py.int<0>)
    %l = makeDict (%zero hash(%hash) : %arg0)
    %1 = test.random
    cf.cond_br %1, ^bb0, ^bb1

^bb0:
    dict_setItem %l[%zero hash(%hash)] to %0
    cf.br ^bb1

^bb1:
    %2 = dict_tryGetItem %l[%zero hash(%hash)]
    %3 = dict_len %l
    return %2, %3 : !py.dynamic, index
}

// CHECK-LABEL: py.func @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[H:.*]] = constant(#py.str<"Hello">)
// CHECK: %[[SIZE:.*]] = arith.constant 1
// CHECK: cf.cond_br %{{.*}}, ^[[COND:.*]], ^[[RET:.*]](%[[ARG0]], %[[SIZE]] : !py.dynamic, index)

// CHECK: ^[[COND]]:
// CHECK-NEXT: %[[IS_UNBOUND:.*]] = isUnboundValue %[[ARG0]]
// CHECK-NEXT: %[[ONE:.*]] = arith.constant 1
// CHECK-NEXT: %[[INCREMENTED:.*]] = arith.addi %[[SIZE]], %[[ONE]]
// CHECK-NEXT: %[[NEW_SIZE:.*]] = arith.select %[[IS_UNBOUND]], %[[INCREMENTED]], %[[SIZE]]
// CHECK-NEXT: cf.br ^[[RET]](%[[H]], %[[NEW_SIZE]] : !py.dynamic, index)

// CHECK: ^[[RET]]
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-SAME: %[[SIZE:[[:alnum:]]+]]
// CHECK-NEXT: return %[[ARG]], %[[SIZE]]

// -----

py.func @test(%arg0 : !py.dynamic, %hash : index) -> !py.dynamic {
    %0 = constant(#py.str<"Hello">)
    %zero = constant(#py.int<0>)
    %one = constant(#py.int<1>)
    %l = makeDict (%zero hash(%hash) : %arg0)
    %2 = dict_tryGetItem %l[%one hash(%hash)]
    return %2 : !py.dynamic
}

// CHECK-LABEL: py.func @test
// CHECK: %[[U:.*]] = constant(#py.unbound)
// CHECK: return %[[U]]

// -----

py.func @test(%arg0 : !py.dynamic, %hash : index) -> (i1, index) {
    %0 = makeDict ()
    %1 = constant(#py.str<"Hello">)
    %r = test.random
    cf.cond_br %r, ^bb1, ^bb2

^bb1:
    dict_setItem %0[%1 hash(%hash)] to %arg0
    cf.br ^bb2

^bb2:
    %res = dict_delItem %1 hash(%hash) from %0
    %s = dict_len %0
    return %res, %s : i1, index
}

// CHECK-LABEL: @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[UNBOUND:.*]] = constant(#py.unbound)
// CHECK-NEXT: %[[SIZE:.*]] = arith.constant 0
// CHECK-NEXT: %[[KEY:.*]] = constant(#py.str<"Hello">)
// CHECK-NEXT: %[[R:.*]] = test.random
// CHECK-NEXT: cf.cond_br %[[R]], ^[[BB1:.*]], ^[[BB2:.*]](%[[SIZE]], %[[UNBOUND]] : index, !py.dynamic)

// CHECK-NEXT: ^[[BB1]]:
// CHECK-NEXT: %[[IS_UNBOUND:.*]] = isUnboundValue %[[UNBOUND]]
// CHECK-NEXT: %[[ONE:.*]] = arith.constant 1
// CHECK-NEXT: %[[INCREMENTED:.*]] = arith.addi %[[SIZE]], %[[ONE]]
// CHECK-NEXT: %[[NEW_SIZE:.*]] = arith.select %[[IS_UNBOUND]], %[[INCREMENTED]], %[[SIZE]]
// CHECK-NEXT: cf.br ^[[BB2]](%[[NEW_SIZE]], %[[ARG0]] : index, !py.dynamic)

// CHECK-NEXT: ^[[BB2]](%[[SIZE:.*]]: index, %[[ARG:.*]]: !py.dynamic):
// CHECK: %[[IS_UNBOUND:.*]] = isUnboundValue %[[ARG]]
// CHECK-NEXT: %[[ONE:.*]] = arith.constant 1
// CHECK-NEXT: %[[TRUE:.*]] = arith.constant true
// CHECK-NEXT: %[[EXISTED:.*]] = arith.xori %[[IS_UNBOUND]], %[[TRUE]]
// CHECK-NEXT: %[[DECREMENTED:.*]] = arith.subi %[[SIZE]], %[[ONE]]
// CHECK-NEXT: %[[NEW_SIZE:.*]] = arith.select %[[IS_UNBOUND]], %[[SIZE]], %[[DECREMENTED]]
// CHECK-NEXT: return %[[EXISTED]], %[[NEW_SIZE]]

// -----

py.func @test(%arg0 : !py.dynamic, %hash : index) -> i1 {
    %0 = makeDict ()
    %1 = constant(#py.int<5>)
    %2 = constant(#py.float<5.0>)
    dict_setItem %0[%1 hash(%hash)] to %arg0
    %res = dict_delItem %2 hash(%hash) from %0
    return %res : i1
}


// CHECK-LABEL: @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[UNBOUND:.*]] = constant(#py.unbound)
// CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0
// CHECK-NEXT: %[[FIVE:.*]] = constant(#py.int<5>)
// CHECK-NEXT: %[[G:.*]] = constant(#py.float<5.{{0+}}e+{{0+}}>)
// CHECK-NEXT: %[[IS_UNBOUND:.*]] = isUnboundValue %[[UNBOUND]]
// CHECK-NEXT: %[[ONE:.*]] = arith.constant 1
// CHECK-NEXT: %[[ADDI:.*]] = arith.addi %[[ZERO]], %[[ONE]]
// CHECK-NEXT: %[[SELECT:.*]] = arith.select %[[IS_UNBOUND]], %[[ADDI]], %[[ZERO]]
// CHECK-NEXT: %[[UNBOUND:.*]] = constant(#py.unbound)
// CHECK-NEXT: %[[IS_UNBOUND:.*]] = isUnboundValue %[[ARG0]]
// CHECK-NEXT: %[[ONE:.*]] = arith.constant 1
// CHECK-NEXT: %[[TRUE:.*]] = arith.constant true
// CHECK-NEXT: %[[INV:.*]] = arith.xori %[[IS_UNBOUND]], %[[TRUE]]
// CHECK-NEXT: %[[SUBI:.*]] = arith.subi %[[SELECT]], %[[ONE]]
// CHECK-NEXT: %[[SELECT2:.*]] = arith.select %[[IS_UNBOUND]], %[[SELECT]], %[[SUBI]]
// CHECK-NEXT: return %[[INV]]
