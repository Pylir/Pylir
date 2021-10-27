// RUN: pylir-opt %s -expand-py-dialect --split-input-file | FileCheck %s

py.globalValue const @one = #py.int<1>

func @linear_search(%tuple : !py.dynamic) -> !py.dynamic {
    %0 = py.getGlobalValue @one
    %1 = py.linearContains %0 in %tuple
    %2 = py.bool.fromI1 %1
    return %2 : !py.dynamic
}

// CHECK-LABEL: @linear_search
// CHECK-SAME: %[[TUPLE:[[:alnum:]]+]]
// CHECK: %[[ONE:.*]] = py.getGlobalValue @one
// CHECK: %[[TUPLE_LEN:.*]] = py.tuple.integer.len %[[TUPLE]]
// CHECK: %[[ZERO:.*]] = constant 0
// CHECK: br ^[[CONDITION:[[:alnum:]]+]]
// CHECK-SAME: %[[ZERO]]

// CHECK: ^[[CONDITION]]
// CHECK-SAME: %[[INDEX:[[:alnum:]]+]]
// CHECK: %[[LESS:.*]] = cmpi ult, %[[INDEX]], %[[TUPLE_LEN]]
// CHECK: %[[FALSE:.*]] = constant false
// CHECK: cond_br %[[LESS]], ^[[BODY:[[:alnum:]]+]], ^[[END:[[:alnum:]]+]]
// CHECK-SAME: %[[FALSE]]

// CHECK: ^[[BODY]]:
// CHECK: %[[ENTRY:.*]] = py.tuple.integer.getItem %[[TUPLE]]
// CHECK-SAME: %[[INDEX]]
// CHECK: %[[IS:.*]] = py.is %[[ENTRY]], %[[ONE]]
// CHECK: %[[TRUE:.*]] = constant true
// CHECK: cond_br %[[IS]], ^[[END]]
// CHECK-SAME: %[[TRUE]]
// CHECK-SAME: ^[[INC:[[:alnum:]]+]]

// CHECK: ^[[INC]]
// CHECK: %[[C1:.*]] = constant 1
// CHECK: %[[NEXT:.*]] = addi %[[INDEX]], %[[C1]]
// CHECK: br ^[[CONDITION]]
// CHECK-SAME: %[[NEXT]]

// CHECK: ^[[END]]
// CHECK-SAME: %[[FOUND:[[:alnum:]]+]]
// CHECK: %[[RET:.*]] = py.bool.fromI1 %[[FOUND]]
// CHECK: return %[[RET]]
