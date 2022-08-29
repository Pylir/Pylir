// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

func.func @test1() -> index {
    %0 = py.constant(#py.int<5>)
    %1 = py.int.toIndex %0
    return %1 : index
}

// CHECK-LABEL: @test1
// CHECK-DAG: %[[C1:.*]] = arith.constant 5
// CHECK-NEXT: return %[[C1]]

func.func @test2(%arg0 : index) -> index {
    %0 = py.int.fromUnsigned %arg0
    %1 = py.int.toIndex %0
    return %1 : index
}

// CHECK-LABEL: @test2
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: return %[[ARG0]]

func.func @test3(%arg0 : index) -> index {
    %0 = py.int.fromSigned %arg0
    %1 = py.int.toIndex %0
    return %1 : index
}

// CHECK-LABEL: @test3
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: return %[[ARG0]]

func.func @test4() -> index {
    %0 = py.constant(#py.int<-5>)
    %1 = py.int.toIndex %0
    return %1 : index
}

// CHECK-LABEL: @test4
// CHECK-DAG: %[[C1:.*]] = arith.constant -5
// CHECK-NEXT: return %[[C1]]


// These are UB, we are just checking it doesn't cause any crashes or so
func.func @test5() -> (index, index) {
    %0 = py.constant(#py.int<523298231467239746239754623792364923764239476239472364>)
    %1 = py.constant(#py.int<-523298231467239746239754623792364923764239476239472364>)
    %2 = py.int.toIndex %0
    %3 = py.int.toIndex %1
    return %2, %3 : index, index
}

// CHECK-LABEL: @test5
