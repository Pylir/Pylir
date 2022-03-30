// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

func @is(%lhs : !py.unknown, %rhs : !py.unknown) -> i1 {
    %0 = py.is %lhs, %rhs : !py.unknown, !py.unknown
    return %0 : i1
}

// CHECK: @is
// CHECK-SAME: %[[LHS:[[:alnum:]]+]]
// CHECK-SAME: %[[RHS:[[:alnum:]]+]]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.icmp "eq" %[[LHS]], %[[RHS]]
// CHECK-NEXT: llvm.return %[[RESULT]]
