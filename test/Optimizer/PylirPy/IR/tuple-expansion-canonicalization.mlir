// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

func @make_tuple_op(%arg0 : !py.unknown) -> !py.unknown {
    %0 = py.makeTuple (%arg0) : (!py.unknown) -> !py.unknown
    %1 = py.constant(#py.int<3>) : !py.unknown
    %2 = py.makeTuple (%1, *%0) : (!py.unknown, !py.unknown) -> !py.unknown
    return %2 : !py.unknown
}

// CHECK-LABEL: @make_tuple_op
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK: %[[CONST:.*]] = py.constant
// CHECK: %[[RESULT:.*]] = py.makeTuple (%[[CONST]], %[[ARG]])
// CHECK: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

func @make_list_op(%arg0 : !py.unknown) -> !py.unknown {
    %0 = py.makeTuple (%arg0) : (!py.unknown) -> !py.unknown
    %1 = py.constant(#py.int<3>) : !py.unknown
    %2 = py.makeList (%1, *%0) : !py.unknown, !py.unknown
    py.return %2 : !py.class<@builtins.list>
}

// CHECK-LABEL: @make_list_op
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK: %[[CONST:.*]] = py.constant
// CHECK: %[[RESULT:.*]] = py.makeList (%[[CONST]], %[[ARG]])
// CHECK: py.return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

func @make_set_op(%arg0 : !py.unknown) -> !py.unknown {
    %0 = py.makeTuple (%arg0) : (!py.unknown) -> !py.unknown
    %1 = py.constant(#py.int<3>) : !py.unknown
    %2 = py.makeSet (%1, *%0) : !py.unknown, !py.unknown
    py.return %2 : !py.class<@builtins.set>
}

// CHECK-LABEL: @make_set_op
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK: %[[CONST:.*]] = py.constant
// CHECK: %[[RESULT:.*]] = py.makeSet (%[[CONST]], %[[ARG]])
// CHECK: py.return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.tuple = #py.type

func @make_tuple_op_constant(%arg0 : !py.unknown) -> !py.unknown {
    %1 = py.constant(#py.tuple<(#py.int<3>, #py.str<"test">)>) : !py.unknown
    %2 = py.makeTuple (%arg0, *%1) : (!py.unknown, !py.unknown) -> !py.unknown
    return %2 : !py.unknown
}

// CHECK-LABEL: @make_tuple_op_constant
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK: %[[CONST1:.*]] = py.constant(#py.int<3>)
// CHECK: %[[CONST2:.*]] = py.constant(#py.str<"test">)
// CHECK: %[[RESULT:.*]] = py.makeTuple (%[[ARG]], %[[CONST1]], %[[CONST2]])
// CHECK: return %[[RESULT]]
