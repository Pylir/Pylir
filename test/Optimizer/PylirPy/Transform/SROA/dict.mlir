// RUN: pylir-opt -pass-pipeline="any(pylir-sroa)" %s --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.int = #py.type

func.func @test(%arg0 : !py.dynamic, %hash : index) -> !py.dynamic {
    %0 = py.constant(#py.str<"Hello">)
    %1 = py.constant(#py.str<" ">)
    %2 = py.constant(#py.str<"World">)
    %zero = py.constant(#py.int<0>)
    %one = py.constant(#py.int<1>)
    %two = py.constant(#py.int<2>)
    %three = py.constant(#py.int<3>)
    %l = py.makeDict (%zero hash(%hash) : %0, %one hash(%hash) : %1, %two hash(%hash) : %2, %three hash(%hash) : %arg0)
    %3 = py.dict.tryGetItem %l[%zero hash(%hash)]
    %4 = py.dict.tryGetItem %l[%one hash(%hash)]
    %5 = py.dict.tryGetItem %l[%two hash(%hash)]
    %6 = py.dict.tryGetItem %l[%three hash(%hash)]
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

func.func @test(%arg0 : !py.dynamic, %hash : index) -> !py.dynamic {
    %0 = py.constant(#py.str<"Hello">)
    %zero = py.constant(#py.int<0>)
    %l = py.makeDict (%zero hash(%hash) : %arg0)
    %1 = test.random
    cf.cond_br %1, ^bb0, ^bb1

^bb0:
    py.dict.setItem %l[%zero hash(%hash)] to %0
    cf.br ^bb1

^bb1:
    %2 = py.dict.tryGetItem %l[%zero hash(%hash)]
    return %2 : !py.dynamic
}

// CHECK-LABEL: func.func @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[H:.*]] = py.constant(#py.str<"Hello">)
// CHECK: cf.cond_br %{{.*}}, ^[[COND:.*]], ^[[RET:.*]](%[[ARG0]] : !py.dynamic)

// CHECK: ^[[COND]]:
// CHECK-NEXT: cf.br ^[[RET]](%[[H]] : !py.dynamic)

// CHECK: ^[[RET]]
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: return %[[ARG]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.int = #py.type

func.func @test(%arg0 : !py.dynamic, %hash : index) -> !py.dynamic {
    %0 = py.constant(#py.str<"Hello">)
    %zero = py.constant(#py.int<0>)
    %one = py.constant(#py.int<1>)
    %l = py.makeDict (%zero hash(%hash) : %arg0)
    %2 = py.dict.tryGetItem %l[%one hash(%hash)]
    return %2 : !py.dynamic
}

// CHECK-LABEL: func.func @test
// CHECK: %[[U:.*]] = py.constant(#py.unbound)
// CHECK: return %[[U]]
