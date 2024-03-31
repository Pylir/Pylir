// RUN: pylir-opt %s --convert-pylirHIR-to-pylirPy --split-input-file | FileCheck %s

pyHIR.globalFunc @basic(%closure, %arg0, %arg1 "first" has_default, *%arg2, %arg3 only "second" has_default, **%arg4) {
  return %arg0
}

// CHECK-LABEL: py.func @basic$impl
// CHECK-NEXT: return

// CHECK-LABEL: py.func @basic(
// CHECK-SAME: %[[CLOSURE:[[:alnum:]]+]]
// CHECK-SAME: %[[TUPLE:[[:alnum:]]+]]
// CHECK-SAME: %[[DICT:[[:alnum:]]+]]

// CHECK: %[[DEFAULT_TUPLE:.*]] = getSlot %[[CLOSURE]]
// CHECK: %[[DEFAULT_DICT:.*]] = getSlot %[[CLOSURE]]
// CHECK: %[[ARG_LEN:.*]] = tuple_len %[[TUPLE]]
// CHECK: %[[UNBOUND:.*]] = constant(#py.unbound)

// %arg0 code:
// CHECK: %[[ZERO:.*]] = arith.constant 0
// CHECK: %[[LT:.*]] = arith.cmpi ult, %[[ZERO]], %[[ARG_LEN]]
// CHECK: cf.cond_br %[[LT]], ^[[BB1:.*]], ^[[BB2:.*]](%[[UNBOUND]] : !py.dynamic)

// CHECK: ^[[BB1]]:
// CHECK: %[[VALUE:.*]] = tuple_getItem %[[TUPLE]][%[[ZERO]]]
// CHECK: cf.br ^[[BB2]](%[[VALUE]] : !py.dynamic)

// CHECK: ^[[BB2]](%[[ARG0:.*]]: !py.dynamic):
// CHECK: %[[IS_UNBOUND:.*]] = isUnboundValue %[[ARG0]]
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: %[[BOUND:.*]] = arith.xori %[[IS_UNBOUND]], %[[TRUE]]
// TODO: This should throw a type error.
// CHECK: cf.assert %[[BOUND]]

// %arg1 code:
// CHECK: %[[ONE:.*]] = arith.constant 1
// CHECK: %[[LT:.*]] = arith.cmpi ult, %[[ONE]], %[[ARG_LEN]]
// CHECK: cf.cond_br %[[LT]], ^[[BB3:.*]], ^[[BB4:.*]](%[[UNBOUND]] : !py.dynamic)

// CHECK: ^[[BB3]]:
// CHECK: %[[VALUE:.*]] = tuple_getItem %[[TUPLE]][%[[ONE]]]
// CHECK: cf.br ^[[BB4]](%[[VALUE]] : !py.dynamic)

// CHECK: ^[[BB4]](%[[ARG1:.*]]: !py.dynamic):
// CHECK: %[[KW:.*]] = constant(#py.str<"first">)
// CHECK: %[[HASH:.*]] = str_hash %[[KW]]
// CHECK: %[[LOOKUP:.*]] = dict_tryGetItem %[[DICT]][%[[KW]] hash(%[[HASH]])]
// CHECK: %[[FAILURE:.*]] = isUnboundValue %[[LOOKUP]]
// CHECK: cf.cond_br %[[FAILURE]], ^[[BB6:.*]](%[[ARG1]] : !py.dynamic), ^[[BB5:.*]]

// CHECK: ^[[BB5]]:
// CHECK: dict_delItem %[[KW]] hash(%[[HASH]]) from %[[DICT]]
// CHECK: cf.br ^[[BB6]]

// CHECK: ^[[BB6]](%[[ARG1:.*]]: !py.dynamic):
// CHECK: %[[FAILURE:.*]] = isUnboundValue %[[ARG1]]
// CHECK: cf.cond_br %[[FAILURE]], ^[[BB7:.*]], ^[[BB8:.*]](%[[ARG1]] : !py.dynamic)

// CHECK: ^[[BB7]]:
// CHECK: %[[ZERO:.*]] = arith.constant 0
// CHECK: %[[DEFAULT:.*]] = tuple_getItem %[[DEFAULT_TUPLE]][%[[ZERO]]]
// CHECK: cf.br ^[[BB8]](%[[DEFAULT]] : !py.dynamic)

// CHECK: ^[[BB8]](%[[ARG1:.*]]: !py.dynamic):
// CHECK: %[[IS_UNBOUND:.*]] = isUnboundValue %[[ARG1]]
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: %[[BOUND:.*]] = arith.xori %[[IS_UNBOUND]], %[[TRUE]]
// TODO: This should throw a type error.
// CHECK: cf.assert %[[BOUND]]

// %arg3 code:
// CHECK: %[[KW:.*]] = constant(#py.str<"second">)
// CHECK: %[[HASH:.*]] = str_hash %[[KW]]
// CHECK: %[[LOOKUP:.*]] = dict_tryGetItem %[[DICT]][%[[KW]] hash(%[[HASH]])]
// CHECK: %[[FAILURE:.*]] = isUnboundValue %[[LOOKUP]]
// CHECK: cf.cond_br %[[FAILURE]], ^[[BB10:.*]](%[[UNBOUND]] : !py.dynamic), ^[[BB9:.*]]

// CHECK: ^[[BB9]]:
// CHECK: dict_delItem %[[KW]] hash(%[[HASH]]) from %[[DICT]]
// CHECK: cf.br ^[[BB10]](%[[LOOKUP]] : !py.dynamic)

// CHECK: ^[[BB10]](%[[ARG3:.*]]: !py.dynamic):
// CHECK: %[[FAILURE:.*]] = isUnboundValue %[[ARG3]]
// CHECK: cf.cond_br %[[FAILURE]], ^[[BB11:.*]], ^[[BB12:.*]](%[[ARG3]] : !py.dynamic)

// CHECK: ^[[BB11]]:
// CHECK: %[[DEFAULT:.*]] = dict_tryGetItem %[[DEFAULT_DICT]][%[[KW]] hash(%[[HASH]])]
// CHECK: cf.br ^[[BB12]](%[[DEFAULT]] : !py.dynamic)

// rest code:
// CHECK: ^[[BB12]](%[[ARG3:.*]]: !py.dynamic):
// CHECK: %[[TWO:.*]] = arith.constant 2
// CHECK: %[[REST:.*]] = tuple_dropFront %[[TWO]], %[[TUPLE]]
// CHECK: %[[RET:.*]] = call @basic$impl(%[[CLOSURE]], %[[ARG0]], %[[ARG1]], %[[REST]], %[[ARG3]], %[[DICT]])
// CHECK: return %[[RET]]
