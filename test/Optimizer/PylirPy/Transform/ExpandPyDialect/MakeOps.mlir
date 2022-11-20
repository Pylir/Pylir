// RUN: pylir-opt %s -p 'builtin.module(any(pylir-expand-py-dialect))' --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.dict = #py.type
py.globalValue @builtins.TypeError =  #py.type
py.globalValue @builtins.None = #py.type
py.globalValue @builtins.function =  #py.type
py.globalValue @builtins.StopIteration = #py.type
py.globalValue @builtins.iter = #py.type
py.globalValue @builtins.next = #py.type

func.func private @pylir__call__(!py.dynamic, !py.dynamic, !py.dynamic) ->  !py.dynamic

func.func @make_list_op(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.constant(#py.int<3>)
    %1 = py.constant(#py.int<4>)
    %2 = py.makeList (%0, *%arg0, %1)
    return %2 : !py.dynamic
}

// CHECK-LABEL: @make_list_op
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[THREE:.*]] = py.constant(#py.int<3>)
// CHECK: %[[FOUR:.*]] = py.constant(#py.int<4>)
// CHECK: %[[ONE:.*]] = arith.constant 1
// CHECK: %[[LIST:.*]] = py.makeList (%[[THREE]])
// CHECK: %[[ITER_F:.*]] = py.constant(#py.ref<@builtins.iter>)
// CHECK: %[[ARGS:.*]] = py.makeTuple (%[[ARG0]])
// CHECK: %[[DICT:.*]] = py.constant(#py.dict<{}>)
// CHECK: %[[ITER:.*]] = py.call @pylir__call__(%[[ITER_F]], %[[ARGS]], %[[DICT]])
// CHECK: cf.br ^[[COND:[[:alnum:]]+]]

// CHECK: ^[[COND]]:
// CHECK: %[[NEXT_F:.*]] = py.constant(#py.ref<@builtins.next>)
// CHECK: %[[ARGS:.*]] = py.makeTuple (%[[ITER]])
// CHECK: %[[DICT:.*]] = py.constant(#py.dict<{}>)
// CHECK: %[[ITEM:.*]] = py.invoke @pylir__call__(%[[NEXT_F]], %[[ARGS]], %[[DICT]])
// CHECK-NEXT: label ^[[BODY:.*]] unwind ^[[EXIT:[[:alnum:]]+]]

// CHECK: ^[[BODY]]:
// CHECK: %[[LEN:.*]] = py.list.len %[[LIST]]
// CHECK: %[[INC:.*]] = arith.addi %[[LEN]], %[[ONE]]
// CHECK: py.list.resize %[[LIST]] to %[[INC]]
// CHECK: py.list.setItem %[[LIST]][%[[LEN]]] to %[[ITEM]]
// CHECK: cf.br ^[[COND]]

// CHECK: ^[[EXIT]](%[[EXC:.*]]: !py.dynamic):
// CHECK: %[[STOP_ITER:.*]] = py.constant(#py.ref<@builtins.StopIteration>)
// CHECK: %[[EXC_TYPE:.*]] = py.typeOf %[[EXC]]
// CHECK: %[[IS:.*]] = py.is %[[STOP_ITER]], %[[EXC_TYPE]]
// CHECK: cf.cond_br %[[IS]], ^[[END:.*]], ^[[RERAISE:[[:alnum:]]+]]

// CHECK: ^[[RERAISE]]:
// CHECK: py.raise %[[EXC]]

// CHECK: ^[[END]]:
// CHECK: %[[LEN:.*]] = py.list.len %[[LIST]]
// CHECK: %[[INC:.*]] = arith.addi %[[LEN]], %[[ONE]]
// CHECK: py.list.resize %[[LIST]] to %[[INC]]
// CHECK: py.list.setItem %[[LIST]][%[[LEN]]] to %[[FOUR]]
// CHECK: return %[[LIST]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.dict = #py.type
py.globalValue @builtins.TypeError =  #py.type
py.globalValue @builtins.None = #py.type
py.globalValue @builtins.function =  #py.type
py.globalValue @builtins.StopIteration = #py.type
py.globalValue @builtins.iter = #py.type
py.globalValue @builtins.next = #py.type

func.func private @pylir__call__(!py.dynamic, !py.dynamic, !py.dynamic) ->  !py.dynamic

func.func @make_tuple_op(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.constant(#py.int<3>)
    %1 = py.constant(#py.int<4>)
    %2 = py.makeTuple (%0, *%arg0, %1)
    return %2 : !py.dynamic
}

// CHECK-LABEL: @make_tuple_op
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[THREE:.*]] = py.constant(#py.int<3>)
// CHECK: %[[FOUR:.*]] = py.constant(#py.int<4>)
// CHECK: %[[ONE:.*]] = arith.constant 1
// CHECK: %[[LIST:.*]] = py.makeList (%[[THREE]])
// CHECK: %[[ITER_F:.*]] = py.constant(#py.ref<@builtins.iter>)
// CHECK: %[[ARGS:.*]] = py.makeTuple (%[[ARG0]])
// CHECK: %[[DICT:.*]] = py.constant(#py.dict<{}>)
// CHECK: %[[ITER:.*]] = py.call @pylir__call__(%[[ITER_F]], %[[ARGS]], %[[DICT]])
// CHECK: cf.br ^[[COND:[[:alnum:]]+]]

// CHECK: ^[[COND]]:
// CHECK: %[[NEXT_F:.*]] = py.constant(#py.ref<@builtins.next>)
// CHECK: %[[ARGS:.*]] = py.makeTuple (%[[ITER]])
// CHECK: %[[DICT:.*]] = py.constant(#py.dict<{}>)
// CHECK: %[[ITEM:.*]] = py.invoke @pylir__call__(%[[NEXT_F]], %[[ARGS]], %[[DICT]])
// CHECK-NEXT: label ^[[BODY:.*]] unwind ^[[EXIT:[[:alnum:]]+]]

// CHECK: ^[[BODY]]:
// CHECK: %[[LEN:.*]] = py.list.len %[[LIST]]
// CHECK: %[[INC:.*]] = arith.addi %[[LEN]], %[[ONE]]
// CHECK: py.list.resize %[[LIST]] to %[[INC]]
// CHECK: py.list.setItem %[[LIST]][%[[LEN]]] to %[[ITEM]]
// CHECK: cf.br ^[[COND]]

// CHECK: ^[[EXIT]](%[[EXC:.*]]: !py.dynamic):
// CHECK: %[[STOP_ITER:.*]] = py.constant(#py.ref<@builtins.StopIteration>)
// CHECK: %[[EXC_TYPE:.*]] = py.typeOf %[[EXC]]
// CHECK: %[[IS:.*]] = py.is %[[STOP_ITER]], %[[EXC_TYPE]]
// CHECK: cf.cond_br %[[IS]], ^[[END:.*]], ^[[RERAISE:[[:alnum:]]+]]

// CHECK: ^[[RERAISE]]:
// CHECK: py.raise %[[EXC]]

// CHECK: ^[[END]]:
// CHECK: %[[LEN:.*]] = py.list.len %[[LIST]]
// CHECK: %[[INC:.*]] = arith.addi %[[LEN]], %[[ONE]]
// CHECK: py.list.resize %[[LIST]] to %[[INC]]
// CHECK: py.list.setItem %[[LIST]][%[[LEN]]] to %[[FOUR]]
// CHECK: %[[TUPLE:.*]] = py.list.toTuple %[[LIST]]
// CHECK: return %[[TUPLE]]

