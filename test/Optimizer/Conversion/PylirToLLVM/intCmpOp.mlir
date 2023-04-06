// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.func @test_eq(%lhs : !py.dynamic, %rhs : !py.dynamic) -> i1 {
    %0 = int_cmp eq %lhs, %rhs
    return %0 : i1
}

// CHECK-LABEL: @test_eq
// CHECK-SAME: %[[LHS:[[:alnum:]]+]]
// CHECK-SAME: %[[RHS:[[:alnum:]]+]]
// CHECK-NEXT: %[[LHS_MPINT:.*]] = llvm.getelementptr %[[LHS]][0, 1]
// CHECK-NEXT: %[[RHS_MPINT:.*]] = llvm.getelementptr %[[RHS]][0, 1]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.call @mp_cmp(%[[LHS_MPINT]], %[[RHS_MPINT]])
// CHECK-NEXT: %[[C:.*]] = llvm.mlir.constant(0 : i{{.*}})
// CHECK-NEXT: %[[CMP:.*]] = llvm.icmp "eq" %[[RESULT]], %[[C]]
// CHECK-NEXT: llvm.return %[[CMP]]

py.func @test_ne(%lhs : !py.dynamic, %rhs : !py.dynamic) -> i1 {
    %0 = int_cmp ne %lhs, %rhs
    return %0 : i1
}

// CHECK-LABEL: @test_ne
// CHECK-SAME: %[[LHS:[[:alnum:]]+]]
// CHECK-SAME: %[[RHS:[[:alnum:]]+]]
// CHECK-NEXT: %[[LHS_MPINT:.*]] = llvm.getelementptr %[[LHS]][0, 1]
// CHECK-NEXT: %[[RHS_MPINT:.*]] = llvm.getelementptr %[[RHS]][0, 1]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.call @mp_cmp(%[[LHS_MPINT]], %[[RHS_MPINT]])
// CHECK-NEXT: %[[C:.*]] = llvm.mlir.constant(0 : i{{.*}})
// CHECK-NEXT: %[[CMP:.*]] = llvm.icmp "ne" %[[RESULT]], %[[C]]
// CHECK-NEXT: llvm.return %[[CMP]]

py.func @test_lt(%lhs : !py.dynamic, %rhs : !py.dynamic) -> i1 {
    %0 = int_cmp lt %lhs, %rhs
    return %0 : i1
}

// CHECK-LABEL: @test_lt
// CHECK-SAME: %[[LHS:[[:alnum:]]+]]
// CHECK-SAME: %[[RHS:[[:alnum:]]+]]
// CHECK-NEXT: %[[LHS_MPINT:.*]] = llvm.getelementptr %[[LHS]][0, 1]
// CHECK-NEXT: %[[RHS_MPINT:.*]] = llvm.getelementptr %[[RHS]][0, 1]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.call @mp_cmp(%[[LHS_MPINT]], %[[RHS_MPINT]])
// CHECK-NEXT: %[[C:.*]] = llvm.mlir.constant(-1 : i{{.*}})
// CHECK-NEXT: %[[CMP:.*]] = llvm.icmp "eq" %[[RESULT]], %[[C]]
// CHECK-NEXT: llvm.return %[[CMP]]

py.func @test_le(%lhs : !py.dynamic, %rhs : !py.dynamic) -> i1 {
    %0 = int_cmp le %lhs, %rhs
    return %0 : i1
}

// CHECK-LABEL: @test_le
// CHECK-SAME: %[[LHS:[[:alnum:]]+]]
// CHECK-SAME: %[[RHS:[[:alnum:]]+]]
// CHECK-NEXT: %[[LHS_MPINT:.*]] = llvm.getelementptr %[[LHS]][0, 1]
// CHECK-NEXT: %[[RHS_MPINT:.*]] = llvm.getelementptr %[[RHS]][0, 1]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.call @mp_cmp(%[[LHS_MPINT]], %[[RHS_MPINT]])
// CHECK-NEXT: %[[C:.*]] = llvm.mlir.constant(1 : i{{.*}})
// CHECK-NEXT: %[[CMP:.*]] = llvm.icmp "ne" %[[RESULT]], %[[C]]
// CHECK-NEXT: llvm.return %[[CMP]]

py.func @test_gt(%lhs : !py.dynamic, %rhs : !py.dynamic) -> i1 {
    %0 = int_cmp gt %lhs, %rhs
    return %0 : i1
}

// CHECK-LABEL: @test_gt
// CHECK-SAME: %[[LHS:[[:alnum:]]+]]
// CHECK-SAME: %[[RHS:[[:alnum:]]+]]
// CHECK-NEXT: %[[LHS_MPINT:.*]] = llvm.getelementptr %[[LHS]][0, 1]
// CHECK-NEXT: %[[RHS_MPINT:.*]] = llvm.getelementptr %[[RHS]][0, 1]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.call @mp_cmp(%[[LHS_MPINT]], %[[RHS_MPINT]])
// CHECK-NEXT: %[[C:.*]] = llvm.mlir.constant(1 : i{{.*}})
// CHECK-NEXT: %[[CMP:.*]] = llvm.icmp "eq" %[[RESULT]], %[[C]]
// CHECK-NEXT: llvm.return %[[CMP]]

py.func @test_ge(%lhs : !py.dynamic, %rhs : !py.dynamic) -> i1 {
    %0 = int_cmp ge %lhs, %rhs
    return %0 : i1
}

// CHECK-LABEL: @test_ge
// CHECK-SAME: %[[LHS:[[:alnum:]]+]]
// CHECK-SAME: %[[RHS:[[:alnum:]]+]]
// CHECK-NEXT: %[[LHS_MPINT:.*]] = llvm.getelementptr %[[LHS]][0, 1]
// CHECK-NEXT: %[[RHS_MPINT:.*]] = llvm.getelementptr %[[RHS]][0, 1]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.call @mp_cmp(%[[LHS_MPINT]], %[[RHS_MPINT]])
// CHECK-NEXT: %[[C:.*]] = llvm.mlir.constant(-1 : i{{.*}})
// CHECK-NEXT: %[[CMP:.*]] = llvm.icmp "ne" %[[RESULT]], %[[C]]
// CHECK-NEXT: llvm.return %[[CMP]]
