// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.BaseException = #py.type

func @make_tuple_ex_op_unique(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeTuple (%arg0)
    %1 = py.constant #py.int<3>
    %2 = py.makeTupleEx (*%0, %1)
        label ^happy unwind ^failure

^happy:
    return %2 : !py.dynamic

^failure:
    %3 = py.landingPad @builtins.BaseException
    py.landingPad.br ^bb3(%3)

^bb3(%e : !py.dynamic):
    return %e : !py.dynamic
}

// CHECK-LABEL: @make_tuple_ex_op_unique
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[THREE:.*]] = py.constant #py.int<3>
// CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[ARG0]], %[[THREE]])
// CHECK: return %[[TUPLE]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

func @make_tuple_ex_op(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeTuple (%arg0)
    %1 = py.constant #py.int<3>
    %2 = py.makeTupleEx (*%0, %1)
        label ^happy unwind ^failure

^happy:
    return %2 : !py.dynamic

^failure:
    %3 = py.landingPad @builtins.BaseException
    py.landingPad.br ^bb3(%3)

^bb3(%e : !py.dynamic):
    return %e : !py.dynamic
}

// CHECK-LABEL: @make_tuple_ex_op
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[THREE:.*]] = py.constant #py.int<3>
// CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[ARG0]], %[[THREE]])
// CHECK: return %[[TUPLE]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

func @make_list_ex_op_unique(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeTuple (%arg0)
    %1 = py.constant #py.int<3>
    %2 = py.makeListEx (*%0, %1)
        label ^happy unwind ^failure

^happy:
    return %2 : !py.dynamic

^failure:
    %3 = py.landingPad @builtins.BaseException
    py.landingPad.br ^bb3(%3)

^bb3(%e : !py.dynamic):
    return %e : !py.dynamic
}

// CHECK-LABEL: @make_list_ex_op_unique
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[THREE:.*]] = py.constant #py.int<3>
// CHECK: %[[LIST:.*]] = py.makeList (%[[ARG0]], %[[THREE]])
// CHECK: return %[[LIST]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

func @make_list_ex_op(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeTuple (%arg0)
    %1 = py.constant #py.int<3>
    %2 = py.makeListEx (*%0, %1)
        label ^happy unwind ^failure

^happy:
    return %2 : !py.dynamic

^failure:
    %3 = py.landingPad @builtins.BaseException
    py.landingPad.br ^bb3(%3)

^bb3(%e : !py.dynamic):
    return %e : !py.dynamic
}

// CHECK-LABEL: @make_list_ex_op
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[THREE:.*]] = py.constant #py.int<3>
// CHECK: %[[LIST:.*]] = py.makeList (%[[ARG0]], %[[THREE]])
// CHECK: return %[[LIST]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

func @make_set_ex_op_unique(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeTuple (%arg0)
    %1 = py.constant #py.int<3>
    %2 = py.makeSetEx (*%0, %1)
        label ^happy unwind ^failure

^happy:
    return %2 : !py.dynamic

^failure:
    %3 = py.landingPad @builtins.BaseException
    py.landingPad.br ^bb3(%3)

^bb3(%e : !py.dynamic):
    return %e : !py.dynamic
}

// CHECK-LABEL: @make_set_ex_op_unique
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[THREE:.*]] = py.constant #py.int<3>
// CHECK: %[[SET:.*]] = py.makeSet (%[[ARG0]], %[[THREE]])
// CHECK: return %[[SET]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

func @make_set_ex_op(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeTuple (%arg0)
    %1 = py.constant #py.int<3>
    %2 = py.makeSetEx (*%0, %1)
        label ^happy unwind ^failure

^happy:
    return %2 : !py.dynamic

^failure:
    %3 = py.landingPad @builtins.BaseException
    py.landingPad.br ^bb3(%3)

^bb3(%e : !py.dynamic):
    return %e : !py.dynamic
}

// CHECK-LABEL: @make_set_ex_op
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[THREE:.*]] = py.constant #py.int<3>
// CHECK: %[[SET:.*]] = py.makeSet (%[[ARG0]], %[[THREE]])
// CHECK: return %[[SET]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

func @make_dict_ex_op_unique(%arg0 : !py.dynamic) -> !py.dynamic {
    %1 = py.constant #py.int<3>
    %2 = py.makeDictEx (%1 : %arg0)
        label ^happy unwind ^failure

^happy:
    return %2 : !py.dynamic

^failure:
    %3 = py.landingPad @builtins.BaseException
    py.landingPad.br ^bb3(%3)

^bb3(%e : !py.dynamic):
    return %e : !py.dynamic
}

// CHECK-LABEL: @make_dict_ex_op_unique
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[THREE:.*]] = py.constant #py.int<3>
// CHECK: %[[DICT:.*]] = py.makeDict (%[[THREE]] : %[[ARG0]])
// CHECK: return %[[DICT]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

func @make_dict_ex_op(%arg0 : !py.dynamic) -> !py.dynamic {
    %1 = py.constant #py.int<3>
    %2 = py.makeDictEx (%1 : %arg0)
        label ^happy unwind ^failure

^happy:
    return %2 : !py.dynamic

^failure:
    %3 = py.landingPad @builtins.BaseException
    py.landingPad.br ^bb3(%3)

^bb3(%e : !py.dynamic):
    return %e : !py.dynamic
}

// CHECK-LABEL: @make_dict_ex_op
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[THREE:.*]] = py.constant #py.int<3>
// CHECK: %[[DICT:.*]] = py.makeDict (%[[THREE]] : %[[ARG0]])
// CHECK: return %[[DICT]]

// -----
