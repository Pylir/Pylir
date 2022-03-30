// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

func @test_eq(%lhs : !py.unknown, %rhs : !py.unknown) -> i1 {
    %0 = py.int.cmp eq %lhs, %rhs : !py.unknown, !py.unknown
    return %0 : i1
}

// CHECK-LABEL: @test_eq
// CHECK-SAME: %[[LHS:[[:alnum:]]+]]
// CHECK-SAME: %[[RHS:[[:alnum:]]+]]
// CHECK-NEXT: %[[LHS_BITCAST:.*]] = llvm.bitcast %[[LHS]]
// CHECK-NEXT: %[[LHS_MPINT:.*]] = llvm.getelementptr %[[LHS_BITCAST]][0, 1]
// CHECK-NEXT: %[[RHS_BITCAST:.*]] = llvm.bitcast %[[RHS]]
// CHECK-NEXT: %[[RHS_MPINT:.*]] = llvm.getelementptr %[[RHS_BITCAST]][0, 1]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.call @mp_cmp(%[[LHS_MPINT]], %[[RHS_MPINT]])
// CHECK-NEXT: %[[C:.*]] = llvm.mlir.constant(0 : i{{.*}})
// CHECK-NEXT: %[[CMP:.*]] = llvm.icmp "eq" %[[RESULT]], %[[C]]
// CHECK-NEXT: llvm.return %[[CMP]]

func @test_ne(%lhs : !py.unknown, %rhs : !py.unknown) -> i1 {
    %0 = py.int.cmp ne %lhs, %rhs : !py.unknown, !py.unknown
    return %0 : i1
}

// CHECK-LABEL: @test_ne
// CHECK-SAME: %[[LHS:[[:alnum:]]+]]
// CHECK-SAME: %[[RHS:[[:alnum:]]+]]
// CHECK-NEXT: %[[LHS_BITCAST:.*]] = llvm.bitcast %[[LHS]]
// CHECK-NEXT: %[[LHS_MPINT:.*]] = llvm.getelementptr %[[LHS_BITCAST]][0, 1]
// CHECK-NEXT: %[[RHS_BITCAST:.*]] = llvm.bitcast %[[RHS]]
// CHECK-NEXT: %[[RHS_MPINT:.*]] = llvm.getelementptr %[[RHS_BITCAST]][0, 1]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.call @mp_cmp(%[[LHS_MPINT]], %[[RHS_MPINT]])
// CHECK-NEXT: %[[C:.*]] = llvm.mlir.constant(0 : i{{.*}})
// CHECK-NEXT: %[[CMP:.*]] = llvm.icmp "ne" %[[RESULT]], %[[C]]
// CHECK-NEXT: llvm.return %[[CMP]]

func @test_lt(%lhs : !py.unknown, %rhs : !py.unknown) -> i1 {
    %0 = py.int.cmp lt %lhs, %rhs : !py.unknown, !py.unknown
    return %0 : i1
}

// CHECK-LABEL: @test_lt
// CHECK-SAME: %[[LHS:[[:alnum:]]+]]
// CHECK-SAME: %[[RHS:[[:alnum:]]+]]
// CHECK-NEXT: %[[LHS_BITCAST:.*]] = llvm.bitcast %[[LHS]]
// CHECK-NEXT: %[[LHS_MPINT:.*]] = llvm.getelementptr %[[LHS_BITCAST]][0, 1]
// CHECK-NEXT: %[[RHS_BITCAST:.*]] = llvm.bitcast %[[RHS]]
// CHECK-NEXT: %[[RHS_MPINT:.*]] = llvm.getelementptr %[[RHS_BITCAST]][0, 1]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.call @mp_cmp(%[[LHS_MPINT]], %[[RHS_MPINT]])
// CHECK-NEXT: %[[C:.*]] = llvm.mlir.constant(-1 : i{{.*}})
// CHECK-NEXT: %[[CMP:.*]] = llvm.icmp "eq" %[[RESULT]], %[[C]]
// CHECK-NEXT: llvm.return %[[CMP]]

func @test_le(%lhs : !py.unknown, %rhs : !py.unknown) -> i1 {
    %0 = py.int.cmp le %lhs, %rhs : !py.unknown, !py.unknown
    return %0 : i1
}

// CHECK-LABEL: @test_le
// CHECK-SAME: %[[LHS:[[:alnum:]]+]]
// CHECK-SAME: %[[RHS:[[:alnum:]]+]]
// CHECK-NEXT: %[[LHS_BITCAST:.*]] = llvm.bitcast %[[LHS]]
// CHECK-NEXT: %[[LHS_MPINT:.*]] = llvm.getelementptr %[[LHS_BITCAST]][0, 1]
// CHECK-NEXT: %[[RHS_BITCAST:.*]] = llvm.bitcast %[[RHS]]
// CHECK-NEXT: %[[RHS_MPINT:.*]] = llvm.getelementptr %[[RHS_BITCAST]][0, 1]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.call @mp_cmp(%[[LHS_MPINT]], %[[RHS_MPINT]])
// CHECK-NEXT: %[[C:.*]] = llvm.mlir.constant(1 : i{{.*}})
// CHECK-NEXT: %[[CMP:.*]] = llvm.icmp "ne" %[[RESULT]], %[[C]]
// CHECK-NEXT: llvm.return %[[CMP]]

func @test_gt(%lhs : !py.unknown, %rhs : !py.unknown) -> i1 {
    %0 = py.int.cmp gt %lhs, %rhs : !py.unknown, !py.unknown
    return %0 : i1
}

// CHECK-LABEL: @test_gt
// CHECK-SAME: %[[LHS:[[:alnum:]]+]]
// CHECK-SAME: %[[RHS:[[:alnum:]]+]]
// CHECK-NEXT: %[[LHS_BITCAST:.*]] = llvm.bitcast %[[LHS]]
// CHECK-NEXT: %[[LHS_MPINT:.*]] = llvm.getelementptr %[[LHS_BITCAST]][0, 1]
// CHECK-NEXT: %[[RHS_BITCAST:.*]] = llvm.bitcast %[[RHS]]
// CHECK-NEXT: %[[RHS_MPINT:.*]] = llvm.getelementptr %[[RHS_BITCAST]][0, 1]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.call @mp_cmp(%[[LHS_MPINT]], %[[RHS_MPINT]])
// CHECK-NEXT: %[[C:.*]] = llvm.mlir.constant(1 : i{{.*}})
// CHECK-NEXT: %[[CMP:.*]] = llvm.icmp "eq" %[[RESULT]], %[[C]]
// CHECK-NEXT: llvm.return %[[CMP]]

func @test_ge(%lhs : !py.unknown, %rhs : !py.unknown) -> i1 {
    %0 = py.int.cmp ge %lhs, %rhs : !py.unknown, !py.unknown
    return %0 : i1
}

// CHECK-LABEL: @test_ge
// CHECK-SAME: %[[LHS:[[:alnum:]]+]]
// CHECK-SAME: %[[RHS:[[:alnum:]]+]]
// CHECK-NEXT: %[[LHS_BITCAST:.*]] = llvm.bitcast %[[LHS]]
// CHECK-NEXT: %[[LHS_MPINT:.*]] = llvm.getelementptr %[[LHS_BITCAST]][0, 1]
// CHECK-NEXT: %[[RHS_BITCAST:.*]] = llvm.bitcast %[[RHS]]
// CHECK-NEXT: %[[RHS_MPINT:.*]] = llvm.getelementptr %[[RHS_BITCAST]][0, 1]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.call @mp_cmp(%[[LHS_MPINT]], %[[RHS_MPINT]])
// CHECK-NEXT: %[[C:.*]] = llvm.mlir.constant(-1 : i{{.*}})
// CHECK-NEXT: %[[CMP:.*]] = llvm.icmp "ne" %[[RESULT]], %[[C]]
// CHECK-NEXT: llvm.return %[[CMP]]
