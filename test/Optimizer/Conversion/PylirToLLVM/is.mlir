// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.func @is(%lhs : !py.dynamic, %rhs : !py.dynamic) -> i1 {
    %0 = is %lhs, %rhs
    return %0 : i1
}

// CHECK: @is
// CHECK-SAME: %[[LHS:[[:alnum:]]+]]
// CHECK-SAME: %[[RHS:[[:alnum:]]+]]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.icmp "eq" %[[LHS]], %[[RHS]]
// CHECK-NEXT: llvm.return %[[RESULT]]
