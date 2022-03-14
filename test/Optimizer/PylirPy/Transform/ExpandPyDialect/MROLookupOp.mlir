// RUN: pylir-opt %s -expand-py-dialect --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.function = #py.type<>
py.globalValue @builtins.dict = #py.type<>

func @linear_search(%tuple : !py.dynamic) -> !py.dynamic {
    %0, %1 = py.mroLookup "__call__" in %tuple
    return %0 : !py.dynamic
}

// CHECK-LABEL: @linear_search
// CHECK-SAME: %[[TUPLE:[[:alnum:]]+]]
// CHECK-NEXT: %[[TUPLE_LEN:.*]] = py.tuple.len %[[TUPLE]]
// CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0
// CHECK-NEXT: br ^[[CONDITION:[[:alnum:]]+]]
// CHECK-SAME: %[[ZERO]]

// CHECK-NEXT: ^[[CONDITION]]
// CHECK-SAME: %[[INDEX:[[:alnum:]]+]]

// CHECK-NEXT: %[[LESS:.*]] = arith.cmpi ult, %[[INDEX]], %[[TUPLE_LEN]]
// CHECK-NEXT: %[[UNBOUND:.*]] = py.constant #py.unbound
// CHECK-NEXT: %[[FALSE:.*]] = arith.constant false
// CHECK-NEXT: cond_br %[[LESS]], ^[[BODY:[[:alnum:]]+]], ^[[END:[[:alnum:]]+]]
// CHECK-SAME: %[[UNBOUND]]
// CHECK-SAME: %[[FALSE]]

// CHECK-NEXT: ^[[BODY]]:
// CHECK-NEXT: %[[ENTRY:.*]] = py.tuple.getItem %[[TUPLE]]
// CHECK-SAME: %[[INDEX]]
// CHECK-NEXT: %[[METATYPE:.*]] = py.typeOf %[[ENTRY]]
// CHECK-NEXT: %[[RESULT:.*]] = py.getSlot "__call__" from %[[ENTRY]] : %[[METATYPE]]
// CHECK-NEXT: %[[FAILURE:.*]] = py.isUnboundValue %[[RESULT]]
// CHECK-NEXT: %[[TRUE:.*]] = arith.constant true
// CHECK-NEXT: cond_br %[[FAILURE]], ^[[NOT_FOUND:.*]], ^[[END]]
// CHECK-SAME: %[[RESULT]]
// CHECK-SAME: %[[TRUE]]

// CHECK-NEXT: ^[[NOT_FOUND]]
// CHECK-NEXT: %[[ONE:.*]] = arith.constant 1
// CHECK-NEXT: %[[NEXT:.*]] = arith.addi %[[INDEX]], %[[ONE]]
// CHECK-NEXT: br ^[[CONDITION]]
// CHECK-SAME: %[[NEXT]]

// CHECK-NEXT: ^[[END]]
// CHECK-SAME: %[[RESULT:[[:alnum:]]+]]
// CHECK-NEXT: return %[[RESULT]]
