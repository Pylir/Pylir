// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

func @test(%lhs : !py.unknown, %rhs : !py.unknown) -> i1 {
    %0 = py.object.id %lhs : !py.unknown
    %1 = py.object.id %rhs : !py.unknown
    %2 = arith.cmpi "eq", %0, %1 : index
    return %2 : i1
}

// CHECK-LABEL: @test
// CHECK-SAME: %[[LHS:[[:alnum:]]+]]
// CHECK-SAME: %[[RHS:[[:alnum:]]+]]
// CHECK-NEXT: %[[RESULT:.*]] = py.is %[[LHS]], %[[RHS]]
// CHECK-NEXT: return %[[RESULT]]
