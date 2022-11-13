// RUN: pylir-opt %s -pass-pipeline="any(pylir-sroa)" | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type

func.func @test(%arg : !py.dynamic, %hash : index) {
    %0 = py.makeDict ()
    %c0 = arith.constant 2 : index
    %1 = py.constant(#py.str<"item">)
    cf.br ^bb37(%arg : !py.dynamic)

^bb37(%iter : !py.dynamic):
    %123 = py.dict.delItem %1 hash(%hash) from %0
    py.list.setItem %iter[%c0] to %arg
    cf.br ^bb37(%iter : !py.dynamic)
}

// CHECK-LABEL: @test
// CHECK-NOT: py.makeDict