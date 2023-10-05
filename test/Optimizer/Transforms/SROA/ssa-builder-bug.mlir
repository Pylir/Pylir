// RUN: pylir-opt -pass-pipeline="builtin.module(any(pylir-sroa))" %s --split-input-file | FileCheck %s

py.func @test(%arg0 : !py.dynamic, %hash : index) -> !py.dynamic {
    %zero = constant(#py.int<0>)
    %l = makeDict (%zero hash(%hash) : %arg0)
    cf.br ^bb0

^bb1:
    %1 = dict_tryGetItem %l[%zero hash(%hash)]
    test.use(%1) : !py.dynamic
    cf.br ^bb0

^bb0:
    %0 = dict_tryGetItem %l[%zero hash(%hash)]
    test.use(%0) : !py.dynamic
    dict_setItem %l[%zero hash(%hash)] to %0
    cf.br ^bb1
}

// CHECK-LABEL: py.func @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: test.use(%[[ARG0]])
// CHECK: test.use(%[[ARG0]])
