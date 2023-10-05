// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.func @test1() -> index {
    %0 = constant(#py.int<5>)
    %1 = int_toIndex %0
    return %1 : index
}

// CHECK-LABEL: @test1
// CHECK-DAG: %[[C1:.*]] = arith.constant 5
// CHECK-NEXT: return %[[C1]]

py.func @test2(%arg0 : index) -> index {
    %0 = int_fromUnsigned %arg0
    %1 = int_toIndex %0
    return %1 : index
}

// CHECK-LABEL: @test2
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: return %[[ARG0]]

py.func @test3(%arg0 : index) -> index {
    %0 = int_fromSigned %arg0
    %1 = int_toIndex %0
    return %1 : index
}

// CHECK-LABEL: @test3
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: return %[[ARG0]]

py.func @test4() -> index {
    %0 = constant(#py.int<-5>)
    %1 = int_toIndex %0
    return %1 : index
}

// CHECK-LABEL: @test4
// CHECK-DAG: %[[C1:.*]] = arith.constant -5
// CHECK-NEXT: return %[[C1]]


// These are UB, we are just checking it doesn't cause any crashes or so
py.func @test5() -> (index, index) {
    %0 = constant(#py.int<523298231467239746239754623792364923764239476239472364>)
    %1 = constant(#py.int<-523298231467239746239754623792364923764239476239472364>)
    %2 = int_toIndex %0
    %3 = int_toIndex %1
    return %2, %3 : index, index
}

// CHECK-LABEL: @test5
