// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int

py.func @make_tuple_ex_op_unique(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = makeTuple (%arg0)
    %1 = constant(#py.int<3>)
    %2 = makeTupleEx (*%0, %1)
        label ^happy unwind ^failure

^happy:
    return %2 : !py.dynamic

^failure(%e : !py.dynamic):
    return %e : !py.dynamic
}

// CHECK-LABEL: @make_tuple_ex_op_unique
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[THREE:.*]] = constant(#py.int<3>)
// CHECK: %[[TUPLE:.*]] = makeTuple (%[[ARG0]], %[[THREE]])
// CHECK: return %[[TUPLE]]

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int

py.func @make_tuple_ex_op(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = makeTuple (%arg0)
    %1 = constant(#py.int<3>)
    %2 = makeTupleEx (*%0, %1)
        label ^happy unwind ^failure

^happy:
    return %2 : !py.dynamic

^failure(%e : !py.dynamic):
    return %e : !py.dynamic
}

// CHECK-LABEL: @make_tuple_ex_op
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[THREE:.*]] = constant(#py.int<3>)
// CHECK: %[[TUPLE:.*]] = makeTuple (%[[ARG0]], %[[THREE]])
// CHECK: return %[[TUPLE]]

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int

py.func @make_list_ex_op_unique(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = makeTuple (%arg0)
    %1 = constant(#py.int<3>)
    %2 = makeListEx (*%0, %1)
        label ^happy unwind ^failure

^happy:
    return %2 : !py.dynamic

^failure(%e : !py.dynamic):
    return %e : !py.dynamic
}

// CHECK-LABEL: @make_list_ex_op_unique
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[THREE:.*]] = constant(#py.int<3>)
// CHECK: %[[LIST:.*]] = makeList (%[[ARG0]], %[[THREE]])
// CHECK: return %[[LIST]]

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int

py.func @make_list_ex_op(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = makeTuple (%arg0)
    %1 = constant(#py.int<3>)
    %2 = makeListEx (*%0, %1)
        label ^happy unwind ^failure

^happy:
    return %2 : !py.dynamic

^failure(%e : !py.dynamic):
    return %e : !py.dynamic
}

// CHECK-LABEL: @make_list_ex_op
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[THREE:.*]] = constant(#py.int<3>)
// CHECK: %[[LIST:.*]] = makeList (%[[ARG0]], %[[THREE]])
// CHECK: return %[[LIST]]

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int

py.func @make_set_ex_op_unique(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = makeTuple (%arg0)
    %1 = constant(#py.int<3>)
    %2 = makeSetEx (*%0, %1)
        label ^happy unwind ^failure

^happy:
    return %2 : !py.dynamic

^failure(%e : !py.dynamic):
    return %e : !py.dynamic
}

// CHECK-LABEL: @make_set_ex_op_unique
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[THREE:.*]] = constant(#py.int<3>)
// CHECK: %[[SET:.*]] = makeSet (%[[ARG0]], %[[THREE]])
// CHECK: return %[[SET]]

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int

py.func @make_set_ex_op(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = makeTuple (%arg0)
    %1 = constant(#py.int<3>)
    %2 = makeSetEx (*%0, %1)
        label ^happy unwind ^failure

^happy:
    return %2 : !py.dynamic

^failure(%e : !py.dynamic):
    return %e : !py.dynamic
}

// CHECK-LABEL: @make_set_ex_op
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[THREE:.*]] = constant(#py.int<3>)
// CHECK: %[[SET:.*]] = makeSet (%[[ARG0]], %[[THREE]])
// CHECK: return %[[SET]]

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int

py.func @make_dict_ex_op_unique(%arg0 : !py.dynamic, %hash : index) -> !py.dynamic {
    %1 = constant(#py.int<3>)
    %2 = makeDictEx (%1 hash(%hash) : %arg0)
        label ^happy unwind ^failure

^happy:
    return %2 : !py.dynamic

^failure(%e : !py.dynamic):
    return %e : !py.dynamic
}

// CHECK-LABEL: @make_dict_ex_op_unique
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[HASH:[[:alnum:]]+]]
// CHECK: %[[THREE:.*]] = constant(#py.int<3>)
// CHECK: %[[DICT:.*]] = makeDict (%[[THREE]] hash(%[[HASH]]) : %[[ARG0]])
// CHECK: return %[[DICT]]

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int

py.func @make_dict_ex_op(%arg0 : !py.dynamic, %hash : index) -> !py.dynamic {
    %1 = constant(#py.int<3>)
    %2 = makeDictEx (%1 hash(%hash) : %arg0)
        label ^happy unwind ^failure

^happy:
    return %2 : !py.dynamic

^failure(%e : !py.dynamic):
    return %e : !py.dynamic
}

// CHECK-LABEL: @make_dict_ex_op
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[HASH:[[:alnum:]]+]]
// CHECK: %[[THREE:.*]] = constant(#py.int<3>)
// CHECK: %[[DICT:.*]] = makeDict (%[[THREE]] hash(%[[HASH]]) : %[[ARG0]])
// CHECK: return %[[DICT]]
