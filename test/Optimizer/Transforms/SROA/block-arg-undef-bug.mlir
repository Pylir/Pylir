// RUN: pylir-opt %s -pass-pipeline="builtin.module(any(pylir-sroa))" | FileCheck %s

py.func @test(%arg : !py.dynamic, %hash : index) {
    %0 = makeDict ()
    %c0 = arith.constant 2 : index
    %1 = constant(#py.str<"item">)
    cf.br ^bb37(%arg : !py.dynamic)

^bb37(%iter : !py.dynamic):
    %123 = dict_delItem %1 hash(%hash) from %0
    list_setItem %iter[%c0] to %arg
    cf.br ^bb37(%iter : !py.dynamic)
}

// CHECK-LABEL: @test
// CHECK-NOT: makeDict
