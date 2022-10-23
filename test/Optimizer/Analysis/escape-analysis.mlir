// RUN: pylir-opt %s --test-escape-analysis --split-input-file 2>&1 >/dev/null | FileCheck %s

func.func @test() {
    %0 = py.makeTuple ()
    %1 = py.makeTuple ()
    test.use(%1) : !py.dynamic
    %2 = py.makeTuple ()
    %3 = py.makeTuple ()
    cf.br ^bb1(%2, %3 : !py.dynamic, !py.dynamic)

^bb1(%iter1 : !py.dynamic, %iter2 : !py.dynamic):
    test.use(%iter2) : !py.dynamic
    cf.br ^bb1(%iter1, %iter2 : !py.dynamic, !py.dynamic)
}

// CHECK-LABEL: @test
// CHECK: %[[NO_ESCAPE_1:.*]] = py.makeTuple ()
// CHECK: %[[ESCAPE_1:.*]] = py.makeTuple ()
// CHECK: %[[NO_ESCAPE_2:.*]] = py.makeTuple ()
// CHECK: %[[ESCAPE_2:.*]] = py.makeTuple ()

// CHECK: {{^}}Escapes: %[[ESCAPE_1]], %[[ESCAPE_2]]{{[[:space:]]*$}}