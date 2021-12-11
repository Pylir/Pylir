// RUN: pylir-opt %s -expand-py-dialect --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.bool = #py.type
py.globalValue @one = #py.type

func @linear_search(%tuple : !py.dynamic) -> !py.dynamic {
    %0 = py.constant @one
    %1 = py.linearContains %0 in %tuple
    %2 = py.bool.fromI1 %1
    return %2 : !py.dynamic
}

// CHECK-LABEL: @linear_search
// CHECK-SAME: %[[TUPLE:[[:alnum:]]+]]
// CHECK: %[[ONE:.*]] = py.constant @one
// CHECK: %[[TUPLE_LEN:.*]] = py.tuple.len %[[TUPLE]]
// CHECK: %[[ZERO:.*]] = arith.constant 0
// CHECK: br ^[[CONDITION:[[:alnum:]]+]]
// CHECK-SAME: %[[ZERO]]

// CHECK: ^[[CONDITION]]
// CHECK-SAME: %[[INDEX:[[:alnum:]]+]]
// CHECK: %[[LESS:.*]] = arith.cmpi ult, %[[INDEX]], %[[TUPLE_LEN]]
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: cond_br %[[LESS]], ^[[BODY:[[:alnum:]]+]], ^[[END:[[:alnum:]]+]]
// CHECK-SAME: %[[FALSE]]

// CHECK: ^[[BODY]]:
// CHECK: %[[ENTRY:.*]] = py.tuple.getItem %[[TUPLE]]
// CHECK-SAME: %[[INDEX]]
// CHECK: %[[IS:.*]] = py.is %[[ENTRY]], %[[ONE]]
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: cond_br %[[IS]], ^[[END]]
// CHECK-SAME: %[[TRUE]]
// CHECK-SAME: ^[[INC:[[:alnum:]]+]]

// CHECK: ^[[INC]]
// CHECK: %[[C1:.*]] = arith.constant 1
// CHECK: %[[NEXT:.*]] = arith.addi %[[INDEX]], %[[C1]]
// CHECK: br ^[[CONDITION]]
// CHECK-SAME: %[[NEXT]]

// CHECK: ^[[END]]
// CHECK-SAME: %[[FOUND:[[:alnum:]]+]]
// CHECK: %[[RET:.*]] = py.bool.fromI1 %[[FOUND]]
// CHECK: return %[[RET]]
