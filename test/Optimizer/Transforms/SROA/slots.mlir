// RUN: pylir-opt -pass-pipeline="any(pylir-sroa)" %s --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.int = #py.type

func.func @test(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.constant(#py.str<"Hello">)
    %1 = py.constant(#py.str<" ">)
    %2 = py.constant(#py.str<"World">)
    %c = py.constant(#py.ref<@builtins.type>)
    %l = py.makeObject %c
    py.setSlot "zero" of %l : %c to %0
    py.setSlot "one" of %l : %c to %1
    py.setSlot "two" of %l : %c to %2
    py.setSlot "three" of %l : %c to %arg0
    %3 = py.getSlot "zero" from %l : %c
    %4 = py.getSlot "one" from %l : %c
    %5 = py.getSlot "two" from %l : %c
    %6 = py.getSlot "three" from %l : %c
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
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.int = #py.type

func.func @test_neg(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.typeOf %arg0
    %1 = py.typeOf %0
    %2 = py.getSlot "zero" from %0 : %1
    return %2 : !py.dynamic
}


// CHECK-LABEL: func.func @test_neg
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[TYPE:.*]] = py.typeOf %[[ARG0]]
// CHECK-NEXT: %[[METATYPE:.*]] = py.typeOf %[[TYPE]]
// CHECK-NEXT: %[[SLOT:.*]] = py.getSlot "zero" from %[[TYPE]] : %[[METATYPE]]
// CHECK-NEXT: return %[[SLOT]]
