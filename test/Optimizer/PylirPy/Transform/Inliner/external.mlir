// RUN: pylir-opt %s --pylir-inliner --split-input-file | FileCheck %s

py.func private @indirect(%arg0 : i32) -> i32

py.func @test(%arg0 : i32) -> i32 {
    %3 = call @indirect(%arg0) : (i32) -> i32
    return %3 : i32
}

// CHECK-LABEL: @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[VALUE:.*]] = call @indirect(%[[ARG0]])
// CHECK-NEXT: return %[[VALUE]]

