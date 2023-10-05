// RUN: pylir-opt %s -split-input-file -pass-pipeline='builtin.module(any(pylir-conditionals-implications))' | FileCheck %s

py.func @contradiction(%c : i1) -> i1 {
    cf.cond_br %c, ^bb1, ^bb1

^bb1:
    return %c : i1
}

// CHECK-LABEL: @contradiction
// CHECK-SAME: %[[C:.*]]: i1
// CHECK: return %[[C]]

// -----

py.func @path_sensitivity(%c: i1) -> i1 {
    cf.cond_br %c, ^bb0, ^bb1

^bb0:
    cf.br ^bb2

^bb1:
    cf.br ^bb2

^bb2:
    return %c : i1
}

// CHECK-LABEL: @path_sensitivity
// CHECK-SAME: %[[C:.*]]: i1
// CHECK: return %[[C]]

// -----

py.func @test(%c: i1) -> i1 {
    %0 = arith.constant true
    cf.cond_br %c, ^bb1, ^bb2

^bb1:
    %1 = arith.xori %c, %0 : i1
    cf.cond_br %1, ^bb2, ^bb3

^bb2:
    return %c : i1

^bb3:
    return %0 : i1
}

// CHECK-LABEL: @test
// CHECK-SAME: %[[ARG0:.*]]: i1
// CHECK: %[[C:.*]] = arith.constant true
// CHECK: cf.cond_br %[[ARG0]], ^[[BB1:.*]], ^[[BB2:[[:alnum:]]+]]
// CHECK: ^[[BB1]]:
// CHECK: %[[C_0:.*]] = arith.constant true
// CHECK: %[[XOR:.*]] = arith.xori %[[C_0]], %[[C]]
// CHECK: cf.cond_br %[[XOR]], ^[[BB2]], ^[[BB3:[[:alnum:]]+]]
// CHECK: ^[[BB2]]:
// CHECK: return %[[ARG0]]
// CHECK: ^[[BB3]]:
// CHECK: return %[[C]]

// -----

#builtins_None = #py.globalValue<builtins.None, initializer = #py.type>

py.func @test(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = constant(#builtins_None)
    %1 = is %arg0, %0
    cf.cond_br %1, ^bb1, ^bb2

^bb1:
    return %arg0 : !py.dynamic

^bb2:
    return %arg0 : !py.dynamic
}

// -----

py.func @loop_implications(%arg0 : i1) -> i1 {
    cf.br ^bb1

^bb1:
    test.use(%arg0) : i1
    cf.cond_br %arg0, ^bb1, ^bb2

^bb2:
    return %arg0 : i1
}

// CHECK-LABEL: @loop_implications
// CHECK-SAME: %[[ARG0:.*]]: i1
// CHECK: cf.br ^[[BB1:[[:alnum:]]+]]
// CHECK: ^[[BB1]]:
// CHECK: test.use(%[[ARG0]])
// CHECK: cf.cond_br %[[ARG0]], ^[[BB1]], ^[[BB2:[[:alnum:]]+]]
// CHECK: ^[[BB2]]:
// CHECK: %[[C:.*]] = arith.constant false
// CHECK: return %[[C]]

// -----

py.func @not_implications(%arg0 : i1) -> i1 {
    %0 = arith.constant true
    %1 = arith.xori %arg0, %0 : i1
    cf.cond_br %1, ^bb1, ^bb2

^bb1:
    return %arg0 : i1

^bb2:
    return %arg0 : i1
}

// CHECK-LABEL: @not_implications
// CHECK: cf.cond_br %{{.*}}, ^[[BB1:[[:alnum:]]+]], ^[[BB2:[[:alnum:]]+]]
// CHECK: ^[[BB1]]:
// CHECK: %[[C:.*]] = arith.constant false
// CHECK: return %[[C]]
// CHECK: ^[[BB2]]:
// CHECK: %[[C:.*]] = arith.constant true
// CHECK: return %[[C]]

// -----

py.func @and_implications(%arg0 : i1, %arg1 : i1) -> i1 {
    %1 = arith.andi %arg0, %arg1 : i1
    cf.cond_br %1, ^bb1, ^bb2

^bb1:
    return %arg0 : i1

^bb2:
    return %arg0 : i1
}

// CHECK-LABEL: @and_implications
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]: i1
// CHECK: cf.cond_br %{{.*}}, ^[[BB1:[[:alnum:]]+]], ^[[BB2:[[:alnum:]]+]]
// CHECK: ^[[BB1]]:
// CHECK: %[[C:.*]] = arith.constant true
// CHECK: return %[[C]]
// CHECK: ^[[BB2]]:
// CHECK: return %[[ARG0]]

// -----

py.func @or_implications(%arg0 : i1, %arg1 : i1) -> i1 {
    %1 = arith.ori %arg0, %arg1 : i1
    cf.cond_br %1, ^bb1, ^bb2

^bb1:
    return %arg0 : i1

^bb2:
    return %arg0 : i1
}

// CHECK-LABEL: @or_implications
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]: i1
// CHECK: cf.cond_br %{{.*}}, ^[[BB1:[[:alnum:]]+]], ^[[BB2:[[:alnum:]]+]]
// CHECK: ^[[BB1]]:
// CHECK: return %[[ARG0]]
// CHECK: ^[[BB2]]:
// CHECK: %[[C:.*]] = arith.constant false
// CHECK: return %[[C]]

// -----

py.func @facts_in_loop_body(%arg0 : i1) -> i1 {
    cf.cond_br %arg0, ^bb1, ^bb2

^bb1:
    test.use(%arg0) : i1
    cf.cond_br %arg0, ^bb1, ^bb2

^bb2:
    return %arg0 : i1
}

// CHECK-LABEL: @facts_in_loop_body
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]: i1
// CHECK: cf.cond_br %[[ARG0]], ^[[BB1:[[:alnum:]]+]], ^[[BB2:[[:alnum:]]+]]
// CHECK: ^[[BB1]]:
// CHECK: %[[C:.*]] = arith.constant true
// CHECK: test.use(%[[C]])
// CHECK: %[[C:.*]] = arith.constant true
// CHECK: cf.cond_br %[[C]], ^[[BB1]], ^[[BB2]]
// COM: TODO: This could be optimized to return false. Needs pattern equality definition.
// CHECK: ^[[BB2]]:
// CHECK: return %[[ARG0]]
