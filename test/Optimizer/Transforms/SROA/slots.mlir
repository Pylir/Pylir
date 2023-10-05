// RUN: pylir-opt -pass-pipeline="builtin.module(any(pylir-sroa))" %s --split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>

py.func @test(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = constant(#py.str<"Hello">)
    %1 = constant(#py.str<" ">)
    %2 = constant(#py.str<"World">)
    %c = constant(#builtins_type)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %l = makeObject %c
    setSlot %l[%c0] to %0
    setSlot %l[%c1] to %1
    setSlot %l[%c2] to %2
    setSlot %l[%c3] to %arg0
    %3 = getSlot %l[%c0]
    %4 = getSlot %l[%c1]
    %5 = getSlot %l[%c2]
    %6 = getSlot %l[%c3]
    %7 = str_concat %3, %4, %5, %6
    return %7 : !py.dynamic
}

// CHECK-LABEL: py.func @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-DAG: %[[H:.*]] = constant(#py.str<"Hello">)
// CHECK-DAG: %[[S:.*]] = constant(#py.str<" ">)
// CHECK-DAG: %[[W:.*]] = constant(#py.str<"World">)
// CHECK: %[[R:.*]] = str_concat %[[H]], %[[S]], %[[W]], %[[ARG0]]
// CHECK: return %[[R]]

// -----

py.func @test_neg(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = typeOf %arg0
    %c0= arith.constant 0 : index
    %2 = getSlot %0[%c0]
    return %2 : !py.dynamic
}


// CHECK-LABEL: py.func @test_neg
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[TYPE:.*]] = typeOf %[[ARG0]]
// CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0
// CHECK-NEXT: %[[SLOT:.*]] = getSlot %[[TYPE]][%[[ZERO]]]
// CHECK-NEXT: return %[[SLOT]]
