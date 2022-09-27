// RUN: pylir-opt %s -pass-pipeline='any(pylir-expand-py-dialect)' --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.dict = #py.type
py.globalValue @builtins.StopIteration = #py.type
py.globalValue @builtins.ValueError = #py.type
py.globalValue @builtins.iter = #py.type
py.globalValue @builtins.next = #py.type

func.func private @pylir__call__(!py.dynamic, !py.dynamic, !py.dynamic) ->  !py.dynamic

func.func @test(%iterable : !py.dynamic) -> (!py.dynamic, !py.dynamic, !py.dynamic) {
    %a, %b, %c = py.unpack %iterable : (!py.dynamic), !py.dynamic, (!py.dynamic)
    return %a, %b, %c : !py.dynamic, !py.dynamic, !py.dynamic
}

// CHECK-LABEL: func.func @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[STOP_ITER:.*]] = py.constant(#py.ref<@builtins.StopIteration>)
// CHECK: %[[ITER_F:.*]] = py.constant(#py.ref<@builtins.iter>)
// CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[ARG0]])
// CHECK: %[[DICT:.*]] = py.constant(#py.dict<{}>)
// CHECK: %[[ITER:.*]] = py.call @pylir__call__(%[[ITER_F]], %[[TUPLE]], %[[DICT]])
// CHECK: %[[NEXT_F:.*]] = py.constant(#py.ref<@builtins.next>)
// CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[ITER]])
// CHECK: %[[DICT:.*]] = py.constant(#py.dict<{}>)
// CHECK: %[[A:.*]] = py.invoke @pylir__call__(%[[NEXT_F]], %[[TUPLE]], %[[DICT]])
// CHECK-NEXT: label ^[[CONTINUE:.*]] unwind ^[[EXHAUSTED:[[:alnum:]]+]]

// CHECK: ^[[CONTINUE]]:
// CHECK: %[[LIST:.*]] = py.makeList ()
// CHECK: cf.br ^[[REST_ARGS:[[:alnum:]]+]]

// CHECK: ^[[EXHAUSTED]](%[[EXC:.*]]: !py.dynamic):
// CHECK: %[[EXC_TYPE:.*]] = py.typeOf %[[EXC]]
// CHECK: %[[IS_STOP_ITER:.*]] = py.is %[[EXC_TYPE]], %[[STOP_ITER]]
// CHECK: cf.cond_br %[[IS_STOP_ITER]], ^[[VALUE_ERROR_BLOCK:.*]], ^[[RERAISE:[[:alnum:]]+]]

// CHECK: ^[[RERAISE]]:
// CHECK: py.raise %[[EXC]]

// CHECK: ^[[VALUE_ERROR_BLOCK]]:
// CHECK: %[[VALUE_ERROR:.*]] = py.constant(#py.ref<@builtins.ValueError>)
// CHECK: %[[TUPLE:.*]] = py.makeTuple ()
// CHECK: %[[DICT:.*]] = py.constant(#py.dict<{}>)
// CHECK: %[[EXC:.*]] = py.call @pylir__call__(%[[VALUE_ERROR]], %[[TUPLE]], %[[DICT]])
// CHECK: py.raise %[[EXC]]

// CHECK: ^[[REST_ARGS]]:
// CHECK: %[[ONE:.*]] = arith.constant 1
// CHECK: %[[NEXT_F:.*]] = py.constant(#py.ref<@builtins.next>)
// CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[ITER]])
// CHECK: %[[DICT:.*]] = py.constant(#py.dict<{}>)
// CHECK: %[[ELEMENT:.*]] = py.invoke @pylir__call__(%[[NEXT_F]], %[[TUPLE]], %[[DICT]])
// CHECK-NEXT: label ^[[BODY:.*]] unwind ^[[EXHAUSTED:[[:alnum:]]+]]

// CHECK: ^[[BODY]]:
// CHECK: %[[LEN:.*]] = py.list.len %[[LIST]]
// CHECK: %[[NEW_LEN:.*]] = arith.addi %[[LEN]], %[[ONE]]
// CHECK: py.list.resize %[[LIST]] to %[[NEW_LEN]]
// CHECK: py.list.setItem %[[LIST]][%[[LEN]]] to %[[ELEMENT]]
// CHECK: cf.br ^[[REST_ARGS]]

// CHECK: ^[[EXHAUSTED]](%[[EXC:.*]]: !py.dynamic):
// CHECK: %[[STOP_ITER:.*]] = py.constant(#py.ref<@builtins.StopIteration>)
// CHECK: %[[EXC_TYPE:.*]] = py.typeOf %[[EXC]]
// CHECK: %[[IS_STOP_ITER:.*]] = py.is %[[STOP_ITER]], %[[EXC_TYPE]]
// CHECK: cf.cond_br %[[IS_STOP_ITER]], ^[[EXHAUSTED:.*]], ^[[RERAISE:[[:alnum:]]+]]

// CHECK: ^[[RERAISE]]:
// CHECK: py.raise %[[EXC]]

// CHECK: ^[[EXHAUSTED]]:
// CHECK: %[[LIST_LEN:.*]] = py.list.len %[[LIST]]
// CHECK: %[[ONE:.*]] = arith.constant 1
// CHECK: %[[TOO_FEW:.*]] = arith.cmpi ult, %[[LIST_LEN]], %[[ONE]]
// CHECK: cf.cond_br %[[TOO_FEW]], ^[[VALUE_ERROR_BLOCK]], ^[[CONTINUE:[[:alnum:]]+]]

// CHECK: ^[[CONTINUE]]:
// CHECK: %[[ONE_2:.*]] = arith.constant 1
// CHECK: %[[INDEX:.*]] = arith.subi %[[LIST_LEN]], %[[ONE_2]]
// CHECK: %[[C:.*]] = py.list.getItem %[[LIST]][%[[INDEX]]]
// CHECK: %[[NEW_LEN:.*]] = arith.subi %[[LIST_LEN]], %[[ONE]]
// CHECK: py.list.resize %[[LIST]] to %[[NEW_LEN]]
// CHECK: return %[[A]], %[[LIST]], %[[C]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.dict = #py.type
py.globalValue @builtins.StopIteration = #py.type
py.globalValue @builtins.ValueError = #py.type
py.globalValue @builtins.iter = #py.type
py.globalValue @builtins.next = #py.type

func.func private @pylir__call__(!py.dynamic, !py.dynamic, !py.dynamic) ->  !py.dynamic

func.func @no_rest_arg(%iterable : !py.dynamic) -> (!py.dynamic, !py.dynamic, !py.dynamic) {
    %a, %b, %c = py.unpack %iterable : (!py.dynamic, !py.dynamic, !py.dynamic)
    return %a, %b, %c : !py.dynamic, !py.dynamic, !py.dynamic
}

// CHECK-LABEL: func.func @no_rest_arg
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[STOP_ITER:.*]] = py.constant(#py.ref<@builtins.StopIteration>)
// CHECK: %[[ITER_F:.*]] = py.constant(#py.ref<@builtins.iter>)
// CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[ARG0]])
// CHECK: %[[DICT:.*]] = py.constant(#py.dict<{}>)
// CHECK: %[[ITER:.*]] = py.call @pylir__call__(%[[ITER_F]], %[[TUPLE]], %[[DICT]])
// CHECK: %[[NEXT_F:.*]] = py.constant(#py.ref<@builtins.next>)
// CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[ITER]])
// CHECK: %[[DICT:.*]] = py.constant(#py.dict<{}>)
// CHECK: %[[A:.*]] = py.invoke @pylir__call__(%[[NEXT_F]], %[[TUPLE]], %[[DICT]])
// CHECK-NEXT: label ^[[CONTINUE:.*]] unwind ^[[EXHAUSTED:[[:alnum:]]+]]

// CHECK: ^[[CONTINUE]]:
// CHECK: %[[NEXT_F:.*]] = py.constant(#py.ref<@builtins.next>)
// CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[ITER]])
// CHECK: %[[DICT:.*]] = py.constant(#py.dict<{}>)
// CHECK: %[[B:.*]] = py.invoke @pylir__call__(%[[NEXT_F]], %[[TUPLE]], %[[DICT]])
// CHECK-NEXT: label ^[[CONTINUE:.*]] unwind ^[[EXHAUSTED:[[:alnum:]]+]]

// CHECK: ^[[CONTINUE]]:
// CHECK: %[[NEXT_F:.*]] = py.constant(#py.ref<@builtins.next>)
// CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[ITER]])
// CHECK: %[[DICT:.*]] = py.constant(#py.dict<{}>)
// CHECK: %[[C:.*]] = py.invoke @pylir__call__(%[[NEXT_F]], %[[TUPLE]], %[[DICT]])
// CHECK-NEXT: label ^[[CONTINUE:.*]] unwind ^[[EXHAUSTED:[[:alnum:]]+]]

// CHECK: ^[[CONTINUE]]:
// CHECK: %[[NEXT_F:.*]] = py.constant(#py.ref<@builtins.next>)
// CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[ITER]])
// CHECK: %[[DICT:.*]] = py.constant(#py.dict<{}>)
// CHECK: py.invoke @pylir__call__(%[[NEXT_F]], %[[TUPLE]], %[[DICT]])
// CHECK-NEXT: label ^[[NOT_EXHAUSTED:.*]] unwind ^[[SHOULD_BE_EXHAUSTED:[[:alnum:]]+]]

// CHECK: ^[[EXHAUSTED]](%[[EXC:.*]]: !py.dynamic):
// CHECK: %[[EXC_TYPE:.*]] = py.typeOf %[[EXC]]
// CHECK: %[[IS_STOP_ITER:.*]] = py.is %[[EXC_TYPE]], %[[STOP_ITER]]
// CHECK: cf.cond_br %[[IS_STOP_ITER]], ^[[VALUE_ERROR_BLOCK:.*]], ^[[RERAISE:[[:alnum:]]+]]

// CHECK: ^[[RERAISE]]:
// CHECK: py.raise %[[EXC]]

// CHECK: ^[[VALUE_ERROR_BLOCK]]:
// CHECK: %[[VALUE_ERROR:.*]] = py.constant(#py.ref<@builtins.ValueError>)
// CHECK: %[[TUPLE:.*]] = py.makeTuple ()
// CHECK: %[[DICT:.*]] = py.constant(#py.dict<{}>)
// CHECK: %[[EXC:.*]] = py.call @pylir__call__(%[[VALUE_ERROR]], %[[TUPLE]], %[[DICT]])
// CHECK: py.raise %[[EXC]]

// CHECK: ^[[NOT_EXHAUSTED]]:
// CHECK: cf.br ^[[VALUE_ERROR_BLOCK]]

// CHECK: ^[[SHOULD_BE_EXHAUSTED]](%[[EXC:.*]]: !py.dynamic):
// CHECK: %[[EXC_TYPE:.*]] = py.typeOf %[[EXC]]
// CHECK: %[[IS_STOP_ITER:.*]] = py.is %[[EXC_TYPE]], %[[STOP_ITER]]
// CHECK: cf.cond_br %[[IS_STOP_ITER]], ^[[CONTINUE:.*]], ^[[RERAISE:[[:alnum:]]+]]

// CHECK: ^[[RERAISE]]:
// CHECK: py.raise %[[EXC]]

// CHECK: ^[[CONTINUE]]:
// CHECK: return %[[A]], %[[B]], %[[C]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.dict = #py.type
py.globalValue @builtins.StopIteration = #py.type
py.globalValue @builtins.ValueError = #py.type
py.globalValue @builtins.iter = #py.type
py.globalValue @builtins.next = #py.type

func.func private @pylir__call__(!py.dynamic, !py.dynamic, !py.dynamic) ->  !py.dynamic

func.func @test_exception(%iterable : !py.dynamic) -> (!py.dynamic, !py.dynamic, !py.dynamic) {
    %a, %b, %c = py.unpackEx %iterable : (!py.dynamic), !py.dynamic, (!py.dynamic)
        label ^ret unwind ^error

^ret:
    return %a, %b, %c : !py.dynamic, !py.dynamic, !py.dynamic

^error(%e: !py.dynamic):
    py.raise %e
}

// CHECK-LABEL: func.func @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[STOP_ITER:.*]] = py.constant(#py.ref<@builtins.StopIteration>)
// CHECK: %[[ITER_F:.*]] = py.constant(#py.ref<@builtins.iter>)
// CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[ARG0]])
// CHECK: %[[DICT:.*]] = py.constant(#py.dict<{}>)
// CHECK: %[[ITER:.*]] = py.invoke @pylir__call__(%[[ITER_F]], %[[TUPLE]], %[[DICT]])
// CHECK-NEXT: label ^[[CONTINUE:.*]] unwind ^[[ERROR:[[:alnum:]]+]]

// CHECK: ^[[CONTINUE]]:
// CHECK: %[[NEXT_F:.*]] = py.constant(#py.ref<@builtins.next>)
// CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[ITER]])
// CHECK: %[[DICT:.*]] = py.constant(#py.dict<{}>)
// CHECK: %[[A:.*]] = py.invoke @pylir__call__(%[[NEXT_F]], %[[TUPLE]], %[[DICT]])
// CHECK-NEXT: label ^[[CONTINUE:.*]] unwind ^[[EXHAUSTED:[[:alnum:]]+]]

// CHECK: ^[[CONTINUE]]:
// CHECK: %[[LIST:.*]] = py.makeList ()
// CHECK: cf.br ^[[REST_ARGS:[[:alnum:]]+]]

// CHECK: ^[[EXHAUSTED]](%[[EXC:.*]]: !py.dynamic):
// CHECK: %[[EXC_TYPE:.*]] = py.typeOf %[[EXC]]
// CHECK: %[[IS_STOP_ITER:.*]] = py.is %[[EXC_TYPE]], %[[STOP_ITER]]
// CHECK: cf.cond_br %[[IS_STOP_ITER]], ^[[VALUE_ERROR_BLOCK:.*]], ^[[RERAISE:[[:alnum:]]+]]

// CHECK: ^[[RERAISE]]:
// CHECK: cf.br ^[[ERROR]](%[[EXC]] : !py.dynamic)

// CHECK: ^[[VALUE_ERROR_BLOCK]]:
// CHECK: %[[VALUE_ERROR:.*]] = py.constant(#py.ref<@builtins.ValueError>)
// CHECK: %[[TUPLE:.*]] = py.makeTuple ()
// CHECK: %[[DICT:.*]] = py.constant(#py.dict<{}>)
// CHECK: %[[EXC:.*]] = py.invoke @pylir__call__(%[[VALUE_ERROR]], %[[TUPLE]], %[[DICT]])
// CHECK-NEXT: label ^[[CONTINUE:.*]] unwind ^[[ERROR]]

// CHECK: ^[[CONTINUE]]:
// CHECK: cf.br ^[[ERROR]](%[[EXC]] : !py.dynamic)

// CHECK: ^[[REST_ARGS]]:
// CHECK: %[[ONE:.*]] = arith.constant 1
// CHECK: %[[NEXT_F:.*]] = py.constant(#py.ref<@builtins.next>)
// CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[ITER]])
// CHECK: %[[DICT:.*]] = py.constant(#py.dict<{}>)
// CHECK: %[[ELEMENT:.*]] = py.invoke @pylir__call__(%[[NEXT_F]], %[[TUPLE]], %[[DICT]])
// CHECK-NEXT: label ^[[BODY:.*]] unwind ^[[EXHAUSTED:[[:alnum:]]+]]

// CHECK: ^[[BODY]]:
// CHECK: %[[LEN:.*]] = py.list.len %[[LIST]]
// CHECK: %[[NEW_LEN:.*]] = arith.addi %[[LEN]], %[[ONE]]
// CHECK: py.list.resize %[[LIST]] to %[[NEW_LEN]]
// CHECK: py.list.setItem %[[LIST]][%[[LEN]]] to %[[ELEMENT]]
// CHECK: cf.br ^[[REST_ARGS]]

// CHECK: ^[[EXHAUSTED]](%[[EXC:.*]]: !py.dynamic):
// CHECK: %[[STOP_ITER:.*]] = py.constant(#py.ref<@builtins.StopIteration>)
// CHECK: %[[EXC_TYPE:.*]] = py.typeOf %[[EXC]]
// CHECK: %[[IS_STOP_ITER:.*]] = py.is %[[STOP_ITER]], %[[EXC_TYPE]]
// CHECK: cf.cond_br %[[IS_STOP_ITER]], ^[[EXHAUSTED:.*]], ^[[RERAISE:[[:alnum:]]+]]

// CHECK: ^[[RERAISE]]:
// CHECK: cf.br ^[[ERROR]](%[[EXC]] : !py.dynamic)

// CHECK: ^[[EXHAUSTED]]:
// CHECK: %[[LIST_LEN:.*]] = py.list.len %[[LIST]]
// CHECK: %[[ONE:.*]] = arith.constant 1
// CHECK: %[[TOO_FEW:.*]] = arith.cmpi ult, %[[LIST_LEN]], %[[ONE]]
// CHECK: cf.cond_br %[[TOO_FEW]], ^[[VALUE_ERROR_BLOCK]], ^[[CONTINUE:[[:alnum:]]+]]

// CHECK: ^[[CONTINUE]]:
// CHECK: %[[ONE_2:.*]] = arith.constant 1
// CHECK: %[[INDEX:.*]] = arith.subi %[[LIST_LEN]], %[[ONE_2]]
// CHECK: %[[C:.*]] = py.list.getItem %[[LIST]][%[[INDEX]]]
// CHECK: %[[NEW_LEN:.*]] = arith.subi %[[LIST_LEN]], %[[ONE]]
// CHECK: py.list.resize %[[LIST]] to %[[NEW_LEN]]
// CHECK: return %[[A]], %[[LIST]], %[[C]]

// CHECK: ^[[ERROR]](%[[EXC:.*]]: !py.dynamic):
// CHECK: py.raise %[[EXC]]
