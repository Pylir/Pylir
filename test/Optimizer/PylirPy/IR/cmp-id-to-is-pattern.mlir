// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

func.func @test(%lhs : !py.dynamic, %rhs : !py.dynamic) -> i1 {
    %0 = py.object.id %lhs
    %1 = py.object.id %rhs
    %2 = arith.cmpi "eq", %0, %1 : index
    return %2 : i1
}

// CHECK-LABEL: @test
// CHECK-SAME: %[[LHS:[[:alnum:]]+]]
// CHECK-SAME: %[[RHS:[[:alnum:]]+]]
// CHECK-NEXT: %[[RESULT:.*]] = py.is %[[LHS]], %[[RHS]]
// CHECK-NEXT: return %[[RESULT]]

func.func @test2(%lhs : !py.dynamic, %rhs : !py.dynamic) -> i1 {
    %0 = py.object.id %lhs
    %1 = py.object.id %rhs
    %2 = arith.cmpi "ne", %0, %1 : index
    return %2 : i1
}

// CHECK-LABEL: @test2
// CHECK-SAME: %[[LHS:[[:alnum:]]+]]
// CHECK-SAME: %[[RHS:[[:alnum:]]+]]
// CHECK-DAG: %[[C:.*]] = arith.constant true
// CHECK: %[[TEMP:.*]] = py.is %[[LHS]], %[[RHS]]
// CHECK-NEXT: %[[RESULT:.*]] = arith.xori %[[TEMP]], %[[C]]
// CHECK-NEXT: return %[[RESULT]]
