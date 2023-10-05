// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.func @make_tuple_op(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = makeTuple (%arg0)
    %1 = constant(#py.int<3>)
    %2 = makeTuple (%1, *%0)
    return %2 : !py.dynamic
}

// CHECK-LABEL: @make_tuple_op
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK: %[[CONST:.*]] = constant
// CHECK: %[[RESULT:.*]] = makeTuple (%[[CONST]], %[[ARG]])
// CHECK: return %[[RESULT]]

// -----

py.func @make_list_op(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = makeTuple (%arg0)
    %1 = constant(#py.int<3>)
    %2 = makeList (%1, *%0)
    return %2 : !py.dynamic
}

// CHECK-LABEL: @make_list_op
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK: %[[CONST:.*]] = constant
// CHECK: %[[RESULT:.*]] = makeList (%[[CONST]], %[[ARG]])
// CHECK: return %[[RESULT]]

// -----

py.func @make_set_op(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = makeTuple (%arg0)
    %1 = constant(#py.int<3>)
    %2 = makeSet (%1, *%0)
    return %2 : !py.dynamic
}

// CHECK-LABEL: @make_set_op
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK: %[[CONST:.*]] = constant
// CHECK: %[[RESULT:.*]] = makeSet (%[[CONST]], %[[ARG]])
// CHECK: return %[[RESULT]]

// -----

py.func @make_tuple_op_constant(%arg0 : !py.dynamic) -> !py.dynamic {
    %1 = constant(#py.tuple<(#py.int<3>, #py.str<"test">)>)
    %2 = makeTuple (%arg0, *%1)
    return %2 : !py.dynamic
}

// CHECK-LABEL: @make_tuple_op_constant
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK: %[[CONST1:.*]] = constant(#py.int<3>)
// CHECK: %[[CONST2:.*]] = constant(#py.str<"test">)
// CHECK: %[[RESULT:.*]] = makeTuple (%[[ARG]], %[[CONST1]], %[[CONST2]])
// CHECK: return %[[RESULT]]
