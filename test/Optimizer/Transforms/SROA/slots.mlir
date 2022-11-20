// RUN: pylir-opt -pass-pipeline="builtin.module(any(pylir-sroa))" %s --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.int = #py.type

func.func @test(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.constant(#py.str<"Hello">)
    %1 = py.constant(#py.str<" ">)
    %2 = py.constant(#py.str<"World">)
    %c = py.constant(#py.ref<@builtins.type>)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %l = py.makeObject %c
    py.setSlot %l[%c0] to %0
    py.setSlot %l[%c1] to %1
    py.setSlot %l[%c2] to %2
    py.setSlot %l[%c3] to %arg0
    %3 = py.getSlot %l[%c0]
    %4 = py.getSlot %l[%c1]
    %5 = py.getSlot %l[%c2]
    %6 = py.getSlot %l[%c3]
    %7 = py.str.concat %3, %4, %5, %6
    return %7 : !py.dynamic
}

// CHECK-LABEL: func.func @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-DAG: %[[H:.*]] = py.constant(#py.str<"Hello">)
// CHECK-DAG: %[[S:.*]] = py.constant(#py.str<" ">)
// CHECK-DAG: %[[W:.*]] = py.constant(#py.str<"World">)
// CHECK: %[[R:.*]] = py.str.concat %[[H]], %[[S]], %[[W]], %[[ARG0]]
// CHECK: return %[[R]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.int = #py.type

func.func @test_neg(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.typeOf %arg0
    %c0= arith.constant 0 : index
    %2 = py.getSlot %0[%c0]
    return %2 : !py.dynamic
}


// CHECK-LABEL: func.func @test_neg
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[TYPE:.*]] = py.typeOf %[[ARG0]]
// CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0
// CHECK-NEXT: %[[SLOT:.*]] = py.getSlot %[[TYPE]][%[[ZERO]]]
// CHECK-NEXT: return %[[SLOT]]
