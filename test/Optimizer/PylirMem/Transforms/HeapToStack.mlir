// RUN: pylir-opt %s --pylir-heap-to-stack=max-object-size=32 --split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple

// CHECK: #[[$TUPLE:.*]] = #py.globalValue<builtins.tuple{{.*}}>

py.func @test() {
    %c0 = arith.constant 0 : index
    %t = constant(#builtins_tuple)
    %m0 = pyMem.gcAllocObject %t[%c0]
    %0 = pyMem.initTuple %m0 to ()
    %m1 = pyMem.gcAllocObject %t[%c0]
    %1 = pyMem.initTuple %m1 to ()
    test.use(%1) : !py.dynamic
    %m2 = pyMem.gcAllocObject %t[%c0]
    %2 = pyMem.initTuple %m2 to ()
    %m3 = pyMem.gcAllocObject %t[%c0]
    %3 = pyMem.initTuple %m3 to ()
    cf.br ^bb1(%2, %3 : !py.dynamic, !py.dynamic)

^bb1(%iter1 : !py.dynamic, %iter2 : !py.dynamic):
    test.use(%iter2) : !py.dynamic
    cf.br ^bb1(%iter1, %iter2 : !py.dynamic, !py.dynamic)
}

// CHECK-LABEL: @test
// CHECK: %[[C:.*]] = arith.constant 0
// CHECK: %[[T:.*]] = constant(#[[$TUPLE]])
// CHECK: pyMem.stackAllocObject tuple %[[T]][0]
// CHECK: pyMem.gcAllocObject %[[T]][%[[C]]]
// CHECK: pyMem.stackAllocObject tuple %[[T]][0]
// CHECK: pyMem.gcAllocObject %[[T]][%[[C]]]

py.func @too_large() {
    %c = arith.constant 128 : index
    %t = constant(#builtins_tuple)
    %m0 = pyMem.gcAllocObject %t[%c]
    %0 = pyMem.initTuple %m0 to ()
    return
}

// CHECK-LABEL: @too_large
// CHECK: %[[C:.*]] = arith.constant 128
// CHECK: %[[T:.*]] = constant(#[[$TUPLE]])
// CHECK: pyMem.gcAllocObject %[[T]][%[[C]]]
