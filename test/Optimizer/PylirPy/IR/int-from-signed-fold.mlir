// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

func.func @test() -> !py.dynamic {
    %0 = arith.constant 5 : index
    %1 = py.int.fromSigned %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-NEXT: %[[C:.*]] = py.constant(#py.int<5>)
// CHECK-NEXT: return %[[C]]

func.func @test2() -> !py.dynamic {
    %0 = arith.constant -5 : index
    %1 = py.int.fromSigned %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @test2
// CHECK-NEXT: %[[C:.*]] = py.constant(#py.int<-5>)
// CHECK-NEXT: return %[[C]]

func.func @test3(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.int.toIndex %arg0
    %1 = py.int.fromSigned %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @test3
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: return %[[ARG0]]
