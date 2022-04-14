// Outlandish growth number to make it inline either of them inside of each other
// This test currently just checks that it does not hang basically
// RUN: pylir-opt %s --pylir-inliner='max-func-growth=500' --split-input-file

func @test(%arg0 : i32) -> i32 {
    %0 = arith.constant 0 : i32
    %1 = arith.cmpi eq, %arg0, %0 : i32
    cf.cond_br %1, ^exit, ^tail

^exit:
    return %0 : i32

^tail:
    %3 = call @indirect(%arg0) : (i32) -> i32
    return %3 : i32
}

func @indirect(%arg0 : i32) -> i32 {
    %0 = arith.constant 1 : i32
    %1 = arith.subi %arg0, %0 : i32
    %2 = call @test(%1) : (i32) -> i32
    return %2 : i32
}
