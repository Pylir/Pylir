// RUN: pylir-opt %s --pylir-inliner='max-inlining-iterations=4' --split-input-file | FileCheck %s

func.func @indirect(%arg0 : i32) -> i32 {
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
