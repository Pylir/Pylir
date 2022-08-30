// RUN: pylir-opt -pass-pipeline="any(pylir-sroa)" %s --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.int = #py.type

func.func @test(%arg0 : !py.dynamic, %hash : index) -> !py.dynamic {
    %zero = py.constant(#py.int<0>)
    %l = py.makeDict (%zero hash(%hash) : %arg0)
    cf.br ^bb0

^bb1:
    %1 = py.dict.tryGetItem %l[%zero hash(%hash)]
    test.use(%1) : !py.dynamic
    cf.br ^bb0

^bb0:
    %0 = py.dict.tryGetItem %l[%zero hash(%hash)]
    test.use(%0) : !py.dynamic
    py.dict.setItem %l[%zero hash(%hash)] to %0
    cf.br ^bb1
}

// CHECK-LABEL: func.func @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: test.use(%[[ARG0]])
// CHECK: test.use(%[[ARG0]])
