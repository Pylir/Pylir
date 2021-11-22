// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

func @make_tuple_op(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeTuple (%arg0)
    %1 = py.constant #py.int<3>
    %2 = py.makeTuple (%1, *%0)
    return %2 : !py.dynamic
}

// CHECK-LABEL: @make_tuple_op
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK: %[[CONST:.*]] = py.constant
// CHECK: %[[RESULT:.*]] = py.makeTuple (%[[CONST]], %[[ARG]])
// CHECK: return %[[RESULT]]

// -----

func @make_list_op(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeTuple (%arg0)
    %1 = py.constant #py.int<3>
    %2 = py.makeList (%1, *%0)
    return %2 : !py.dynamic
}

// CHECK-LABEL: @make_list_op
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK: %[[CONST:.*]] = py.constant
// CHECK: %[[RESULT:.*]] = py.makeList (%[[CONST]], %[[ARG]])
// CHECK: return %[[RESULT]]

// -----

func @make_set_op(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeTuple (%arg0)
    %1 = py.constant #py.int<3>
    %2 = py.makeSet (%1, *%0)
    return %2 : !py.dynamic
}

// CHECK-LABEL: @make_set_op
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK: %[[CONST:.*]] = py.constant
// CHECK: %[[RESULT:.*]] = py.makeSet (%[[CONST]], %[[ARG]])
// CHECK: return %[[RESULT]]

// -----

func @make_tuple_op_constant(%arg0 : !py.dynamic) -> !py.dynamic {
    %1 = py.constant #py.tuple<(#py.int<3>, #py.str<"test">)>
    %2 = py.makeTuple (%arg0, *%1)
    return %2 : !py.dynamic
}

// CHECK-LABEL: @make_tuple_op_constant
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK: %[[CONST1:.*]] = py.constant #py.int<3>
// CHECK: %[[CONST2:.*]] = py.constant #py.str<"test">
// CHECK: %[[RESULT:.*]] = py.makeTuple (%[[ARG]], %[[CONST1]], %[[CONST2]])
// CHECK: return %[[RESULT]]
