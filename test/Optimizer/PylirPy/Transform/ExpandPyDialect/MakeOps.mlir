// RUN: pylir-opt %s -p 'builtin.module(any(pylir-expand-py-dialect))' --split-input-file | FileCheck %s

py.func private @pylir__call__(!py.dynamic, !py.dynamic, !py.dynamic) ->  !py.dynamic

py.func @make_list_op(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = constant(#py.int<3>)
    %1 = constant(#py.int<4>)
    %2 = makeList (%0, *%arg0, %1)
    return %2 : !py.dynamic
}

// CHECK-DAG: #[[$STOP:.*]] = #py.globalValue<builtins.StopIteration{{.*}}>
// CHECK-DAG: #[[$ITER:.*]] = #py.globalValue<builtins.iter{{.*}}>
// CHECK-DAG: #[[$NEXT:.*]] = #py.globalValue<builtins.next{{.*}}>

// CHECK-LABEL: @make_list_op
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[THREE:.*]] = constant(#py.int<3>)
// CHECK: %[[FOUR:.*]] = constant(#py.int<4>)
// CHECK: %[[ONE:.*]] = arith.constant 1
// CHECK: %[[LIST:.*]] = makeList (%[[THREE]])
// CHECK: %[[ITER_F:.*]] = constant(#[[$ITER]])
// CHECK: %[[ARGS:.*]] = makeTuple (%[[ARG0]])
// CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
// CHECK: %[[ITER:.*]] = call @pylir__call__(%[[ITER_F]], %[[ARGS]], %[[DICT]])
// CHECK: cf.br ^[[COND:[[:alnum:]]+]]

// CHECK: ^[[COND]]:
// CHECK: %[[NEXT_F:.*]] = constant(#[[$NEXT]])
// CHECK: %[[ARGS:.*]] = makeTuple (%[[ITER]])
// CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
// CHECK: %[[ITEM:.*]] = invoke @pylir__call__(%[[NEXT_F]], %[[ARGS]], %[[DICT]])
// CHECK-NEXT: label ^[[BODY:.*]] unwind ^[[EXIT:[[:alnum:]]+]]

// CHECK: ^[[BODY]]:
// CHECK: %[[LEN:.*]] = list_len %[[LIST]]
// CHECK: %[[INC:.*]] = arith.addi %[[LEN]], %[[ONE]]
// CHECK: list_resize %[[LIST]] to %[[INC]]
// CHECK: list_setItem %[[LIST]][%[[LEN]]] to %[[ITEM]]
// CHECK: cf.br ^[[COND]]

// CHECK: ^[[EXIT]](%[[EXC:.*]]: !py.dynamic):
// CHECK: %[[STOP_ITER:.*]] = constant(#[[$STOP]])
// CHECK: %[[EXC_TYPE:.*]] = typeOf %[[EXC]]
// CHECK: %[[IS:.*]] = is %[[STOP_ITER]], %[[EXC_TYPE]]
// CHECK: cf.cond_br %[[IS]], ^[[END:.*]], ^[[RERAISE:[[:alnum:]]+]]

// CHECK: ^[[RERAISE]]:
// CHECK: raise %[[EXC]]

// CHECK: ^[[END]]:
// CHECK: %[[LEN:.*]] = list_len %[[LIST]]
// CHECK: %[[INC:.*]] = arith.addi %[[LEN]], %[[ONE]]
// CHECK: list_resize %[[LIST]] to %[[INC]]
// CHECK: list_setItem %[[LIST]][%[[LEN]]] to %[[FOUR]]
// CHECK: return %[[LIST]]

// -----

py.func private @pylir__call__(!py.dynamic, !py.dynamic, !py.dynamic) ->  !py.dynamic

py.func @make_tuple_op(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = constant(#py.int<3>)
    %1 = constant(#py.int<4>)
    %2 = makeTuple (%0, *%arg0, %1)
    return %2 : !py.dynamic
}

// CHECK-DAG: #[[$STOP:.*]] = #py.globalValue<builtins.StopIteration{{.*}}>
// CHECK-DAG: #[[$ITER:.*]] = #py.globalValue<builtins.iter{{.*}}>
// CHECK-DAG: #[[$NEXT:.*]] = #py.globalValue<builtins.next{{.*}}>

// CHECK-LABEL: @make_tuple_op
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[THREE:.*]] = constant(#py.int<3>)
// CHECK: %[[FOUR:.*]] = constant(#py.int<4>)
// CHECK: %[[ONE:.*]] = arith.constant 1
// CHECK: %[[LIST:.*]] = makeList (%[[THREE]])
// CHECK: %[[ITER_F:.*]] = constant(#[[$ITER]])
// CHECK: %[[ARGS:.*]] = makeTuple (%[[ARG0]])
// CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
// CHECK: %[[ITER:.*]] = call @pylir__call__(%[[ITER_F]], %[[ARGS]], %[[DICT]])
// CHECK: cf.br ^[[COND:[[:alnum:]]+]]

// CHECK: ^[[COND]]:
// CHECK: %[[NEXT_F:.*]] = constant(#[[$NEXT]])
// CHECK: %[[ARGS:.*]] = makeTuple (%[[ITER]])
// CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
// CHECK: %[[ITEM:.*]] = invoke @pylir__call__(%[[NEXT_F]], %[[ARGS]], %[[DICT]])
// CHECK-NEXT: label ^[[BODY:.*]] unwind ^[[EXIT:[[:alnum:]]+]]

// CHECK: ^[[BODY]]:
// CHECK: %[[LEN:.*]] = list_len %[[LIST]]
// CHECK: %[[INC:.*]] = arith.addi %[[LEN]], %[[ONE]]
// CHECK: list_resize %[[LIST]] to %[[INC]]
// CHECK: list_setItem %[[LIST]][%[[LEN]]] to %[[ITEM]]
// CHECK: cf.br ^[[COND]]

// CHECK: ^[[EXIT]](%[[EXC:.*]]: !py.dynamic):
// CHECK: %[[STOP_ITER:.*]] = constant(#[[$STOP]])
// CHECK: %[[EXC_TYPE:.*]] = typeOf %[[EXC]]
// CHECK: %[[IS:.*]] = is %[[STOP_ITER]], %[[EXC_TYPE]]
// CHECK: cf.cond_br %[[IS]], ^[[END:.*]], ^[[RERAISE:[[:alnum:]]+]]

// CHECK: ^[[RERAISE]]:
// CHECK: raise %[[EXC]]

// CHECK: ^[[END]]:
// CHECK: %[[LEN:.*]] = list_len %[[LIST]]
// CHECK: %[[INC:.*]] = arith.addi %[[LEN]], %[[ONE]]
// CHECK: list_resize %[[LIST]] to %[[INC]]
// CHECK: list_setItem %[[LIST]][%[[LEN]]] to %[[FOUR]]
// CHECK: %[[TUPLE:.*]] = list_toTuple %[[LIST]]
// CHECK: return %[[TUPLE]]

