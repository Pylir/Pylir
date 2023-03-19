// RUN: pylir-opt %s --pylir-inliner='max-inlining-iterations=1 optimization-pipeline=any(canonicalize)' --split-input-file | FileCheck %s

py.func @indirect(%arg0 : i32) -> i32 {
    %0 = arith.constant 1 : i32
    %1 = arith.subi %arg0, %0 : i32
    %2 = call @indirect(%1) : (i32) -> i32
    return %2 : i32
}

// CHECK-LABEL: func @indirect
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = arith.constant 1 : i32
// CHECK-NEXT: %[[SUB:.*]] = arith.subi %[[ARG0]], %[[C]]
// CHECK-NEXT: %[[CALL:.*]] = call @indirect(%[[SUB]])
// CHECK-NEXT: return %[[CALL]]

py.func @recursion_base_case(%arg0 : i32) -> i32 {
    %0 = arith.constant 1 : i32
    %1 = arith.subi %arg0, %0 : i32
    %2 = arith.cmpi slt, %1, %0 : i32
    cf.cond_br %2, ^exit, ^cont

^exit:
    return %0 : i32

^cont:
    %3 = call @recursion_base_case(%0) : (i32) -> i32
    %4 = call @recursion_base_case(%1) : (i32) -> i32
    return %4 : i32
}

// CHECK-LABEL: func @recursion_base_case
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = arith.constant 1 : i32
// CHECK-NEXT: %[[SUB:.*]] = arith.subi %[[ARG0]], %[[C]]
// CHECK-NEXT: %[[COND:.*]] = arith.cmpi slt, %[[SUB]], %[[C]]
// CHECK-NEXT: cf.cond_br %[[COND]], ^[[EXIT:.*]], ^[[CONT:[[:alnum:]]+]]
// CHECK-NEXT: ^[[EXIT]]:
// CHECK-NEXT: return %[[C]]
// CHECK-NEXT: ^[[CONT]]:
// CHECK-NEXT: %[[CALL:.*]] = call @recursion_base_case(%[[SUB]])
// CHECK-NEXT: return %[[CALL]]
