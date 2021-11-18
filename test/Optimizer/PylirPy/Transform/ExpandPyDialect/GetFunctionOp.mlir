// RUN: pylir-opt %s -expand-py-dialect --split-input-file | FileCheck %s

py.globalValue const @builtins.function = #py.int<1>

func @get_function(%callable : !py.dynamic) -> !py.dynamic {
    %0, %1 = py.getFunction %callable
    return %0 : !py.dynamic
}

// CHECK-LABEL: @get_function
// CHECK-SAME: %[[CALLABLE:[[:alnum:]]+]]
// CHECK: %[[FUNCTION:.*]] = py.constant @builtins.function
// CHECK: br ^[[CONDITION:[[:alnum:]]+]]
// CHECK-SAME: %[[CALLABLE]]

// CHECK: ^[[CONDITION]]
// CHECK-SAME: %[[CURRENT:[[:alnum:]]+]]
// CHECK: %[[TYPE:.*]] = py.typeOf %[[CURRENT]]
// CHECK: %[[IS:.*]] = py.is %[[TYPE]], %[[FUNCTION]]
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: cond_br %[[IS]], ^[[END_BLOCK:[[:alnum:]]+]]
// CHECK-SAME: %[[CURRENT]]
// CHECK-SAME: %[[TRUE]]
// CHECK-SAME: ^[[BODY:[[:alnum:]]+]]

// CHECK: ^[[BODY]]
// mro lookup...
// CHECK: %[[RESULT:.*]], %[[SUCCESS:.*]] = py.getAttr "__call__" from %{{[[:alnum:]]+}}
// CHECK: cond_br %[[SUCCESS]], ^[[RESULT_BLOCK:[[:alnum:]]+]]
// CHECK-SAME: %[[RESULT]]
// CHECK-SAME: %[[SUCCESS]]

// CHECK: ^[[RESULT_BLOCK]]
// CHECK-SAME: %[[RESULT:[[:alnum:]]+]]
// CHECK-SAME: %[[SUCCESS:[[:alnum:]]+]]
// CHECK: %[[UNBOUND:.*]] = py.constant #py.unbound
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: cond_br %[[SUCCESS]], ^[[CONDITION]]
// CHECK-SAME: %[[RESULT]]
// CHECK-SAME: ^[[END_BLOCK]]
// CHECK-SAME: %[[UNBOUND]]
// CHECK-SAME: %[[FALSE]]

// CHECK: ^[[END_BLOCK]]
// CHECK-SAME: %[[RESULT:[[:alnum:]]+]]
// CHECK: return %[[RESULT]]
