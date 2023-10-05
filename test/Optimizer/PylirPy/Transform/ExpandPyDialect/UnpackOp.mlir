// RUN: pylir-opt %s -p 'builtin.module(any(pylir-expand-py-dialect))' --split-input-file | FileCheck %s

py.func private @pylir__call__(!py.dynamic, !py.dynamic, !py.dynamic) ->  !py.dynamic

py.func @test(%iterable : !py.dynamic) -> (!py.dynamic, !py.dynamic, !py.dynamic) {
    %a, %b, %c = unpack %iterable : (!py.dynamic), !py.dynamic, (!py.dynamic)
    return %a, %b, %c : !py.dynamic, !py.dynamic, !py.dynamic
}

// CHECK-DAG: #[[$STOP:.*]] = #py.globalValue<builtins.StopIteration{{.*}}>
// CHECK-DAG: #[[$VALUE:.*]] = #py.globalValue<builtins.ValueError{{.*}}>
// CHECK-DAG: #[[$ITER:.*]] = #py.globalValue<builtins.iter{{.*}}>
// CHECK-DAG: #[[$NEXT:.*]] = #py.globalValue<builtins.next{{.*}}>

// CHECK-LABEL: py.func @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[STOP_ITER:.*]] = constant(#[[$STOP]])
// CHECK: %[[ITER_F:.*]] = constant(#[[$ITER]])
// CHECK: %[[TUPLE:.*]] = makeTuple (%[[ARG0]])
// CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
// CHECK: %[[ITER:.*]] = call @pylir__call__(%[[ITER_F]], %[[TUPLE]], %[[DICT]])
// CHECK: %[[NEXT_F:.*]] = constant(#[[$NEXT]])
// CHECK: %[[TUPLE:.*]] = makeTuple (%[[ITER]])
// CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
// CHECK: %[[A:.*]] = invoke @pylir__call__(%[[NEXT_F]], %[[TUPLE]], %[[DICT]])
// CHECK-NEXT: label ^[[CONTINUE:.*]] unwind ^[[EXHAUSTED:[[:alnum:]]+]]

// CHECK: ^[[CONTINUE]]:
// CHECK: %[[LIST:.*]] = makeList ()
// CHECK: cf.br ^[[REST_ARGS:[[:alnum:]]+]]

// CHECK: ^[[EXHAUSTED]](%[[EXC:.*]]: !py.dynamic):
// CHECK: %[[EXC_TYPE:.*]] = typeOf %[[EXC]]
// CHECK: %[[IS_STOP_ITER:.*]] = is %[[EXC_TYPE]], %[[STOP_ITER]]
// CHECK: cf.cond_br %[[IS_STOP_ITER]], ^[[VALUE_ERROR_BLOCK:.*]], ^[[RERAISE:[[:alnum:]]+]]

// CHECK: ^[[RERAISE]]:
// CHECK: raise %[[EXC]]

// CHECK: ^[[VALUE_ERROR_BLOCK]]:
// CHECK: %[[VALUE_ERROR:.*]] = constant(#[[$VALUE]])
// CHECK: %[[TUPLE:.*]] = makeTuple ()
// CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
// CHECK: %[[EXC:.*]] = call @pylir__call__(%[[VALUE_ERROR]], %[[TUPLE]], %[[DICT]])
// CHECK: raise %[[EXC]]

// CHECK: ^[[REST_ARGS]]:
// CHECK: %[[ONE:.*]] = arith.constant 1
// CHECK: %[[NEXT_F:.*]] = constant(#[[$NEXT]])
// CHECK: %[[TUPLE:.*]] = makeTuple (%[[ITER]])
// CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
// CHECK: %[[ELEMENT:.*]] = invoke @pylir__call__(%[[NEXT_F]], %[[TUPLE]], %[[DICT]])
// CHECK-NEXT: label ^[[BODY:.*]] unwind ^[[EXHAUSTED:[[:alnum:]]+]]

// CHECK: ^[[BODY]]:
// CHECK: %[[LEN:.*]] = list_len %[[LIST]]
// CHECK: %[[NEW_LEN:.*]] = arith.addi %[[LEN]], %[[ONE]]
// CHECK: list_resize %[[LIST]] to %[[NEW_LEN]]
// CHECK: list_setItem %[[LIST]][%[[LEN]]] to %[[ELEMENT]]
// CHECK: cf.br ^[[REST_ARGS]]

// CHECK: ^[[EXHAUSTED]](%[[EXC:.*]]: !py.dynamic):
// CHECK: %[[STOP_ITER:.*]] = constant(#[[$STOP]])
// CHECK: %[[EXC_TYPE:.*]] = typeOf %[[EXC]]
// CHECK: %[[IS_STOP_ITER:.*]] = is %[[STOP_ITER]], %[[EXC_TYPE]]
// CHECK: cf.cond_br %[[IS_STOP_ITER]], ^[[EXHAUSTED:.*]], ^[[RERAISE:[[:alnum:]]+]]

// CHECK: ^[[RERAISE]]:
// CHECK: raise %[[EXC]]

// CHECK: ^[[EXHAUSTED]]:
// CHECK: %[[LIST_LEN:.*]] = list_len %[[LIST]]
// CHECK: %[[ONE:.*]] = arith.constant 1
// CHECK: %[[TOO_FEW:.*]] = arith.cmpi ult, %[[LIST_LEN]], %[[ONE]]
// CHECK: cf.cond_br %[[TOO_FEW]], ^[[VALUE_ERROR_BLOCK]], ^[[CONTINUE:[[:alnum:]]+]]

// CHECK: ^[[CONTINUE]]:
// CHECK: %[[ONE_2:.*]] = arith.constant 1
// CHECK: %[[INDEX:.*]] = arith.subi %[[LIST_LEN]], %[[ONE_2]]
// CHECK: %[[C:.*]] = list_getItem %[[LIST]][%[[INDEX]]]
// CHECK: %[[NEW_LEN:.*]] = arith.subi %[[LIST_LEN]], %[[ONE]]
// CHECK: list_resize %[[LIST]] to %[[NEW_LEN]]
// CHECK: return %[[A]], %[[LIST]], %[[C]]

// -----

py.func private @pylir__call__(!py.dynamic, !py.dynamic, !py.dynamic) ->  !py.dynamic

py.func @no_rest_arg(%iterable : !py.dynamic) -> (!py.dynamic, !py.dynamic, !py.dynamic) {
    %a, %b, %c = unpack %iterable : (!py.dynamic, !py.dynamic, !py.dynamic)
    return %a, %b, %c : !py.dynamic, !py.dynamic, !py.dynamic
}

// CHECK-DAG: #[[$STOP:.*]] = #py.globalValue<builtins.StopIteration{{.*}}>
// CHECK-DAG: #[[$VALUE:.*]] = #py.globalValue<builtins.ValueError{{.*}}>
// CHECK-DAG: #[[$ITER:.*]] = #py.globalValue<builtins.iter{{.*}}>
// CHECK-DAG: #[[$NEXT:.*]] = #py.globalValue<builtins.next{{.*}}>

// CHECK-LABEL: py.func @no_rest_arg
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[STOP_ITER:.*]] = constant(#[[$STOP]])
// CHECK: %[[ITER_F:.*]] = constant(#[[$ITER]])
// CHECK: %[[TUPLE:.*]] = makeTuple (%[[ARG0]])
// CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
// CHECK: %[[ITER:.*]] = call @pylir__call__(%[[ITER_F]], %[[TUPLE]], %[[DICT]])
// CHECK: %[[NEXT_F:.*]] = constant(#[[$NEXT]])
// CHECK: %[[TUPLE:.*]] = makeTuple (%[[ITER]])
// CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
// CHECK: %[[A:.*]] = invoke @pylir__call__(%[[NEXT_F]], %[[TUPLE]], %[[DICT]])
// CHECK-NEXT: label ^[[CONTINUE:.*]] unwind ^[[EXHAUSTED:[[:alnum:]]+]]

// CHECK: ^[[CONTINUE]]:
// CHECK: %[[NEXT_F:.*]] = constant(#[[$NEXT]])
// CHECK: %[[TUPLE:.*]] = makeTuple (%[[ITER]])
// CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
// CHECK: %[[B:.*]] = invoke @pylir__call__(%[[NEXT_F]], %[[TUPLE]], %[[DICT]])
// CHECK-NEXT: label ^[[CONTINUE:.*]] unwind ^[[EXHAUSTED:[[:alnum:]]+]]

// CHECK: ^[[CONTINUE]]:
// CHECK: %[[NEXT_F:.*]] = constant(#[[$NEXT]])
// CHECK: %[[TUPLE:.*]] = makeTuple (%[[ITER]])
// CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
// CHECK: %[[C:.*]] = invoke @pylir__call__(%[[NEXT_F]], %[[TUPLE]], %[[DICT]])
// CHECK-NEXT: label ^[[CONTINUE:.*]] unwind ^[[EXHAUSTED:[[:alnum:]]+]]

// CHECK: ^[[CONTINUE]]:
// CHECK: %[[NEXT_F:.*]] = constant(#[[$NEXT]])
// CHECK: %[[TUPLE:.*]] = makeTuple (%[[ITER]])
// CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
// CHECK: invoke @pylir__call__(%[[NEXT_F]], %[[TUPLE]], %[[DICT]])
// CHECK-NEXT: label ^[[NOT_EXHAUSTED:.*]] unwind ^[[SHOULD_BE_EXHAUSTED:[[:alnum:]]+]]

// CHECK: ^[[EXHAUSTED]](%[[EXC:.*]]: !py.dynamic):
// CHECK: %[[EXC_TYPE:.*]] = typeOf %[[EXC]]
// CHECK: %[[IS_STOP_ITER:.*]] = is %[[EXC_TYPE]], %[[STOP_ITER]]
// CHECK: cf.cond_br %[[IS_STOP_ITER]], ^[[VALUE_ERROR_BLOCK:.*]], ^[[RERAISE:[[:alnum:]]+]]

// CHECK: ^[[RERAISE]]:
// CHECK: raise %[[EXC]]

// CHECK: ^[[VALUE_ERROR_BLOCK]]:
// CHECK: %[[VALUE_ERROR:.*]] = constant(#[[$VALUE]])
// CHECK: %[[TUPLE:.*]] = makeTuple ()
// CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
// CHECK: %[[EXC:.*]] = call @pylir__call__(%[[VALUE_ERROR]], %[[TUPLE]], %[[DICT]])
// CHECK: raise %[[EXC]]

// CHECK: ^[[NOT_EXHAUSTED]]:
// CHECK: cf.br ^[[VALUE_ERROR_BLOCK]]

// CHECK: ^[[SHOULD_BE_EXHAUSTED]](%[[EXC:.*]]: !py.dynamic):
// CHECK: %[[EXC_TYPE:.*]] = typeOf %[[EXC]]
// CHECK: %[[IS_STOP_ITER:.*]] = is %[[EXC_TYPE]], %[[STOP_ITER]]
// CHECK: cf.cond_br %[[IS_STOP_ITER]], ^[[CONTINUE:.*]], ^[[RERAISE:[[:alnum:]]+]]

// CHECK: ^[[RERAISE]]:
// CHECK: raise %[[EXC]]

// CHECK: ^[[CONTINUE]]:
// CHECK: return %[[A]], %[[B]], %[[C]]

// -----

py.func private @pylir__call__(!py.dynamic, !py.dynamic, !py.dynamic) ->  !py.dynamic

py.func @test_exception(%iterable : !py.dynamic) -> (!py.dynamic, !py.dynamic, !py.dynamic) {
    %a, %b, %c = unpackEx %iterable : (!py.dynamic), !py.dynamic, (!py.dynamic)
        label ^ret unwind ^error

^ret:
    return %a, %b, %c : !py.dynamic, !py.dynamic, !py.dynamic

^error(%e: !py.dynamic):
    raise %e
}

// CHECK-DAG: #[[$STOP:.*]] = #py.globalValue<builtins.StopIteration{{.*}}>
// CHECK-DAG: #[[$VALUE:.*]] = #py.globalValue<builtins.ValueError{{.*}}>
// CHECK-DAG: #[[$ITER:.*]] = #py.globalValue<builtins.iter{{.*}}>
// CHECK-DAG: #[[$NEXT:.*]] = #py.globalValue<builtins.next{{.*}}>

// CHECK-LABEL: py.func @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[STOP_ITER:.*]] = constant(#[[$STOP]])
// CHECK: %[[ITER_F:.*]] = constant(#[[$ITER]])
// CHECK: %[[TUPLE:.*]] = makeTuple (%[[ARG0]])
// CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
// CHECK: %[[ITER:.*]] = invoke @pylir__call__(%[[ITER_F]], %[[TUPLE]], %[[DICT]])
// CHECK-NEXT: label ^[[CONTINUE:.*]] unwind ^[[ERROR:[[:alnum:]]+]]

// CHECK: ^[[CONTINUE]]:
// CHECK: %[[NEXT_F:.*]] = constant(#[[$NEXT]])
// CHECK: %[[TUPLE:.*]] = makeTuple (%[[ITER]])
// CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
// CHECK: %[[A:.*]] = invoke @pylir__call__(%[[NEXT_F]], %[[TUPLE]], %[[DICT]])
// CHECK-NEXT: label ^[[CONTINUE:.*]] unwind ^[[EXHAUSTED:[[:alnum:]]+]]

// CHECK: ^[[CONTINUE]]:
// CHECK: %[[LIST:.*]] = makeList ()
// CHECK: cf.br ^[[REST_ARGS:[[:alnum:]]+]]

// CHECK: ^[[EXHAUSTED]](%[[EXC:.*]]: !py.dynamic):
// CHECK: %[[EXC_TYPE:.*]] = typeOf %[[EXC]]
// CHECK: %[[IS_STOP_ITER:.*]] = is %[[EXC_TYPE]], %[[STOP_ITER]]
// CHECK: cf.cond_br %[[IS_STOP_ITER]], ^[[VALUE_ERROR_BLOCK:.*]], ^[[RERAISE:[[:alnum:]]+]]

// CHECK: ^[[RERAISE]]:
// CHECK: cf.br ^[[ERROR]](%[[EXC]] : !py.dynamic)

// CHECK: ^[[VALUE_ERROR_BLOCK]]:
// CHECK: %[[VALUE_ERROR:.*]] = constant(#[[$VALUE]])
// CHECK: %[[TUPLE:.*]] = makeTuple ()
// CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
// CHECK: %[[EXC:.*]] = invoke @pylir__call__(%[[VALUE_ERROR]], %[[TUPLE]], %[[DICT]])
// CHECK-NEXT: label ^[[CONTINUE:.*]] unwind ^[[ERROR]]

// CHECK: ^[[CONTINUE]]:
// CHECK: cf.br ^[[ERROR]](%[[EXC]] : !py.dynamic)

// CHECK: ^[[REST_ARGS]]:
// CHECK: %[[ONE:.*]] = arith.constant 1
// CHECK: %[[NEXT_F:.*]] = constant(#[[$NEXT]])
// CHECK: %[[TUPLE:.*]] = makeTuple (%[[ITER]])
// CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
// CHECK: %[[ELEMENT:.*]] = invoke @pylir__call__(%[[NEXT_F]], %[[TUPLE]], %[[DICT]])
// CHECK-NEXT: label ^[[BODY:.*]] unwind ^[[EXHAUSTED:[[:alnum:]]+]]

// CHECK: ^[[BODY]]:
// CHECK: %[[LEN:.*]] = list_len %[[LIST]]
// CHECK: %[[NEW_LEN:.*]] = arith.addi %[[LEN]], %[[ONE]]
// CHECK: list_resize %[[LIST]] to %[[NEW_LEN]]
// CHECK: list_setItem %[[LIST]][%[[LEN]]] to %[[ELEMENT]]
// CHECK: cf.br ^[[REST_ARGS]]

// CHECK: ^[[EXHAUSTED]](%[[EXC:.*]]: !py.dynamic):
// CHECK: %[[STOP_ITER:.*]] = constant(#[[$STOP]])
// CHECK: %[[EXC_TYPE:.*]] = typeOf %[[EXC]]
// CHECK: %[[IS_STOP_ITER:.*]] = is %[[STOP_ITER]], %[[EXC_TYPE]]
// CHECK: cf.cond_br %[[IS_STOP_ITER]], ^[[EXHAUSTED:.*]], ^[[RERAISE:[[:alnum:]]+]]

// CHECK: ^[[RERAISE]]:
// CHECK: cf.br ^[[ERROR]](%[[EXC]] : !py.dynamic)

// CHECK: ^[[EXHAUSTED]]:
// CHECK: %[[LIST_LEN:.*]] = list_len %[[LIST]]
// CHECK: %[[ONE:.*]] = arith.constant 1
// CHECK: %[[TOO_FEW:.*]] = arith.cmpi ult, %[[LIST_LEN]], %[[ONE]]
// CHECK: cf.cond_br %[[TOO_FEW]], ^[[VALUE_ERROR_BLOCK]], ^[[CONTINUE:[[:alnum:]]+]]

// CHECK: ^[[CONTINUE]]:
// CHECK: %[[ONE_2:.*]] = arith.constant 1
// CHECK: %[[INDEX:.*]] = arith.subi %[[LIST_LEN]], %[[ONE_2]]
// CHECK: %[[C:.*]] = list_getItem %[[LIST]][%[[INDEX]]]
// CHECK: %[[NEW_LEN:.*]] = arith.subi %[[LIST_LEN]], %[[ONE]]
// CHECK: list_resize %[[LIST]] to %[[NEW_LEN]]
// CHECK: return %[[A]], %[[LIST]], %[[C]]

// CHECK: ^[[ERROR]](%[[EXC:.*]]: !py.dynamic):
// CHECK: raise %[[EXC]]
