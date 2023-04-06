// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.func @strEqual(%lhs : !py.dynamic, %rhs : !py.dynamic) -> i1 {
    %0 = str_equal %lhs, %rhs
    return %0 : i1
}

// CHECK: @strEqual
// CHECK-SAME: %[[LHS:[[:alnum:]]+]]
// CHECK-SAME: %[[RHS:[[:alnum:]]+]]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.icmp "eq" %[[LHS]], %[[RHS]]
// CHECK-NEXT: llvm.cond_br %[[RESULT]], ^[[EXIT:[[:alnum:]]+]](%[[RESULT]] : i1), ^[[LEN_CHECK:[[:alnum:]]+]]
// CHECK-NEXT: ^[[LEN_CHECK]]
// CHECK-NEXT: %[[LHS_BUFFER:.*]] = llvm.getelementptr %[[LHS]][0, 1]
// CHECK-NEXT: %[[RHS_BUFFER:.*]] = llvm.getelementptr %[[RHS]][0, 1]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[LHS_BUFFER]][0, 0]
// CHECK-NEXT: %[[LHS_LEN:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[RHS_BUFFER]][0, 0]
// CHECK-NEXT: %[[RHS_LEN:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[LEN_EQUAL:.*]] = llvm.icmp "eq" %[[LHS_LEN]], %[[RHS_LEN]]
// CHECK-NEXT: llvm.cond_br %[[LEN_EQUAL]], ^[[NOT_ZERO_CHECK:[[:alnum:]]+]], ^[[EXIT]](%[[LEN_EQUAL]] : i1)
// CHECK-NEXT: ^[[NOT_ZERO_CHECK]]
// CHECK-NEXT: %[[ZERO_I:.*]] = llvm.mlir.constant(0 : index)
// CHECK-NEXT: %[[IS_ZERO:.*]] = llvm.icmp "eq" %[[LHS_LEN]], %[[ZERO_I]]
// CHECK-NEXT: llvm.cond_br %[[IS_ZERO]], ^[[EXIT]](%[[IS_ZERO]] : i1), ^[[CONTENT_CHECK:[[:alnum:]]+]]
// CHECK-NEXT: ^[[CONTENT_CHECK]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[LHS_BUFFER]][0, 2]
// CHECK-NEXT: %[[LHS_CHAR:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[RHS_BUFFER]][0, 2]
// CHECK-NEXT: %[[RHS_CHAR:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.call @memcmp(%[[LHS_CHAR]], %[[RHS_CHAR]], %[[LHS_LEN]])
// CHECK-NEXT: %[[ZERO_I:.*]] = llvm.mlir.constant(0 : i32)
// CHECK-NEXT: %[[EQUAL:.*]] = llvm.icmp "eq" %[[RESULT]], %[[ZERO_I]]
// CHECK-NEXT: llvm.br ^[[EXIT]](%[[EQUAL]] : i1)
// CHECK-NEXT: ^[[EXIT]](%[[RESULT:.*]]: i1):
// CHECK-NEXT: llvm.return %[[RESULT]]
