// RUN: pylir-opt -pass-pipeline="any(pylir-sroa)" %s --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.int = #py.type

func.func @test(%arg0 : !py.dynamic, %hash : index) -> !py.dynamic {
    %0 = py.constant(#py.str<"Hello">)
    %1 = py.constant(#py.str<" ">)
    %2 = py.constant(#py.str<"World">)
    %zero = py.constant(#py.int<0>)
    %one = py.constant(#py.int<1>)
    %two = py.constant(#py.int<2>)
    %three = py.constant(#py.int<3>)
    %l = py.makeDict (%zero hash(%hash) : %0, %one hash(%hash) : %1, %two hash(%hash) : %2, %three hash(%hash) : %arg0)
    %3 = py.dict.tryGetItem %l[%zero hash(%hash)]
    %4 = py.dict.tryGetItem %l[%one hash(%hash)]
    %5 = py.dict.tryGetItem %l[%two hash(%hash)]
    %6 = py.dict.tryGetItem %l[%three hash(%hash)]
    %7 = py.str.concat %3, %4, %5, %6
    return %7 : !py.dynamic
}

// CHECK-LABEL: func.func @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-DAG: %[[H:.*]] = py.constant(#py.str<"Hello">)
// CHECK-DAG: %[[S:.*]] = py.constant(#py.str<" ">)
// CHECK-DAG: %[[W:.*]] = py.constant(#py.str<"World">)
// CHECK: %[[R:.*]] = py.str.concat %[[H]], %[[S]], %[[W]], %[[ARG0]]
// CHECK: return %[[R]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.int = #py.type

func.func @test(%arg0 : !py.dynamic, %hash : index) -> (!py.dynamic, index) {
    %0 = py.constant(#py.str<"Hello">)
    %zero = py.constant(#py.int<0>)
    %l = py.makeDict (%zero hash(%hash) : %arg0)
    %1 = test.random
    cf.cond_br %1, ^bb0, ^bb1

^bb0:
    py.dict.setItem %l[%zero hash(%hash)] to %0
    cf.br ^bb1

^bb1:
    %2 = py.dict.tryGetItem %l[%zero hash(%hash)]
    %3 = py.dict.len %l
    return %2, %3 : !py.dynamic, index
}

// CHECK-LABEL: func.func @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[H:.*]] = py.constant(#py.str<"Hello">)
// CHECK: %[[SIZE:.*]] = arith.constant 1
// CHECK: cf.cond_br %{{.*}}, ^[[COND:.*]], ^[[RET:.*]](%[[ARG0]], %[[SIZE]] : !py.dynamic, index)

// CHECK: ^[[COND]]:
// CHECK-NEXT: %[[IS_UNBOUND:.*]] = py.isUnboundValue %[[ARG0]]
// CHECK-NEXT: %[[ONE:.*]] = arith.constant 1
// CHECK-NEXT: %[[INCREMENTED:.*]] = arith.addi %[[SIZE]], %[[ONE]]
// CHECK-NEXT: %[[NEW_SIZE:.*]] = arith.select %[[IS_UNBOUND]], %[[INCREMENTED]], %[[SIZE]]
// CHECK-NEXT: cf.br ^[[RET]](%[[H]], %[[NEW_SIZE]] : !py.dynamic, index)

// CHECK: ^[[RET]]
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-SAME: %[[SIZE:[[:alnum:]]+]]
// CHECK-NEXT: return %[[ARG]], %[[SIZE]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.int = #py.type

func.func @test(%arg0 : !py.dynamic, %hash : index) -> !py.dynamic {
    %0 = py.constant(#py.str<"Hello">)
    %zero = py.constant(#py.int<0>)
    %one = py.constant(#py.int<1>)
    %l = py.makeDict (%zero hash(%hash) : %arg0)
    %2 = py.dict.tryGetItem %l[%one hash(%hash)]
    return %2 : !py.dynamic
}

// CHECK-LABEL: func.func @test
// CHECK: %[[U:.*]] = py.constant(#py.unbound)
// CHECK: return %[[U]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type

func.func @test(%arg0 : !py.dynamic, %hash : index) -> (i1, index) {
    %0 = py.makeDict ()
    %1 = py.constant(#py.str<"Hello">)
    %r = test.random
    cf.cond_br %r, ^bb1, ^bb2

^bb1:
    py.dict.setItem %0[%1 hash(%hash)] to %arg0
    cf.br ^bb2

^bb2:
    %res = py.dict.delItem %1 hash(%hash) from %0
    %s = py.dict.len %0
    return %res, %s : i1, index
}

// CHECK-LABEL: @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[UNBOUND:.*]] = py.constant(#py.unbound)
// CHECK-NEXT: %[[SIZE:.*]] = arith.constant 0
// CHECK-NEXT: %[[KEY:.*]] = py.constant(#py.str<"Hello">)
// CHECK-NEXT: %[[R:.*]] = test.random
// CHECK-NEXT: cf.cond_br %[[R]], ^[[BB1:.*]], ^[[BB2:.*]](%[[SIZE]], %[[UNBOUND]] : index, !py.dynamic)

// CHECK-NEXT: ^[[BB1]]:
// CHECK-NEXT: %[[IS_UNBOUND:.*]] = py.isUnboundValue %[[UNBOUND]]
// CHECK-NEXT: %[[ONE:.*]] = arith.constant 1
// CHECK-NEXT: %[[INCREMENTED:.*]] = arith.addi %[[SIZE]], %[[ONE]]
// CHECK-NEXT: %[[NEW_SIZE:.*]] = arith.select %[[IS_UNBOUND]], %[[INCREMENTED]], %[[SIZE]]
// CHECK-NEXT: cf.br ^[[BB2]](%[[NEW_SIZE]], %[[ARG0]] : index, !py.dynamic)

// CHECK-NEXT: ^[[BB2]](%[[SIZE:.*]]: index, %[[ARG:.*]]: !py.dynamic):
// CHECK: %[[IS_UNBOUND:.*]] = py.isUnboundValue %[[ARG]]
// CHECK-NEXT: %[[ONE:.*]] = arith.constant 1
// CHECK-NEXT: %[[TRUE:.*]] = arith.constant true
// CHECK-NEXT: %[[EXISTED:.*]] = arith.xori %[[IS_UNBOUND]], %[[TRUE]]
// CHECK-NEXT: %[[DECREMENTED:.*]] = arith.subi %[[SIZE]], %[[ONE]]
// CHECK-NEXT: %[[NEW_SIZE:.*]] = arith.select %[[IS_UNBOUND]], %[[SIZE]], %[[DECREMENTED]]
// CHECK-NEXT: return %[[EXISTED]], %[[NEW_SIZE]]
