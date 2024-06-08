// RUN: pylir-opt %s --split-input-file -pylir-dce

// CHECK-LABEL: func @test
// CHECK-NOT: ^{{.*}}:
py.func @test()  {
    return

^bb1:
    cf.br ^bb1
}
