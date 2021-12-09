// RUN: pylir-opt %s -expand-py-dialect --split-input-file | FileCheck %s

func @linear_search(%tuple : !py.dynamic) -> !py.dynamic {
    %0, %1 = py.mroLookup "__call__" in %tuple
    return %0 : !py.dynamic
}

// CHECK-LABEL: @linear_search
// CHECK-SAME: %[[TUPLE:[[:alnum:]]+]]
// CHECK: %[[TUPLE_LEN:.*]] = py.tuple.len %[[TUPLE]]
// CHECK: %[[ZERO:.*]] = arith.constant 0
// CHECK: br ^[[CONDITION:[[:alnum:]]+]]
// CHECK-SAME: %[[ZERO]]

// CHECK: ^[[CONDITION]]
// CHECK-SAME: %[[INDEX:[[:alnum:]]+]]

// CHECK: %[[LESS:.*]] = arith.cmpi ult, %[[INDEX]], %[[TUPLE_LEN]]
// CHECK: %[[UNBOUND:.*]] = py.constant #py.unbound
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: cond_br %[[LESS]], ^[[BODY:[[:alnum:]]+]], ^[[END:[[:alnum:]]+]]
// CHECK-SAME: %[[UNBOUND]]
// CHECK-SAME: %[[FALSE]]

// CHECK: ^[[BODY]]:
// CHECK: %[[ENTRY:.*]] = py.tuple.getItem %[[TUPLE]]
// CHECK-SAME: %[[INDEX]]
// CHECK: %[[RESULT:.*]] = py.getSlot "__call__" from %[[ENTRY]]
// CHECK: %[[SUCCESS:.*]] = py.isUnboundValue %[[RESULT]]
// CHECK: cond_br %[[SUCCESS]], ^[[END]]
// CHECK-SAME: %[[RESULT]]
// CHECK-SAME: %[[SUCCESS]]
// CHECK-SAME: ^[[NOT_FOUND:[[:alnum:]]+]]

// CHECK: ^[[NOT_FOUND]]
// CHECK: %[[ONE:.*]] = arith.constant 1
// CHECK: %[[NEXT:.*]] = arith.addi %[[INDEX]], %[[ONE]]
// CHECK: br ^[[CONDITION]]
// CHECK-SAME: %[[NEXT]]

// CHECK: ^[[END]]
// CHECK-SAME: %[[RESULT:[[:alnum:]]+]]
// CHECK: return %[[RESULT]]
