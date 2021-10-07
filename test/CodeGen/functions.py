# RUN: pylir %s -emit-mlir -o - | FileCheck %s

# CHECK-LABEL: __init__

# CHECK: %[[RES:.*]] = py.makeFunc @"foo$cc[0]"
# CHECK: %[[NAME:.*]] = py.constant "foo"
# CHECK: py.setAttr "__name__" of %[[RES]] to %[[NAME]]
# CHECK: %[[NAME:.*]] = py.constant "foo"
# CHECK: py.setAttr "__qualname__" of %[[RES]] to %[[NAME]]
# CHECK: %[[DEFAULTS:.*]] = py.getGlobalValue @builtins.None
# CHECK: py.setAttr "__defaults__" of %[[RES]] to %[[DEFAULTS]]
# CHECK: %[[KWDEFAULTS:.*]] = py.getGlobalValue @builtins.None
# CHECK: py.setAttr "__kwdefaults__" of %[[RES]] to %[[KWDEFAULTS]]
# CHECK: %[[CLOSURE:.*]] = py.getGlobalValue @builtins.None
# CHECK: py.setAttr "__closure__" of %[[RES]] to %[[CLOSURE]]
# CHECK: %[[FOO:.*]] = py.getGlobalHandle @foo
# CHECK: py.store %[[RES]] into %[[FOO]]

def foo():
    x = 3

    def bar(a=3, *, c=1):
        return a + c

# CHECK-LABEL: func private @"foo$impl[0]"

# CHECK: %[[THREE:.*]] = py.constant #py.int<3>
# CHECK: py.store %[[THREE]] into %{{.*}}

# CHECK: %[[THREE:.*]] = py.constant #py.int<3>
# CHECK: %[[ONE:.*]] = py.constant #py.int<1>
# CHECK: %[[C:.*]] = py.constant "c"
# CHECK: %[[RES:.*]] = py.makeFunc @"foo.<locals>.bar$cc[0]"
# CHECK: %[[NAME:.*]] = py.constant "bar"
# CHECK: py.setAttr "__name__" of %[[RES]] to %[[NAME]]
# CHECK: %[[NAME:.*]] = py.constant "foo.<locals>.bar"
# CHECK: py.setAttr "__qualname__" of %[[RES]] to %[[NAME]]
# CHECK: %[[DEFAULTS:.*]] = py.makeTuple (%[[THREE]])
# CHECK: py.setAttr "__defaults__" of %[[RES]] to %[[DEFAULTS]]
# CHECK: %[[KWDEFAULTS:.*]] = py.makeDict (%[[C]] : %[[ONE]])
# CHECK: py.setAttr "__kwdefaults__" of %[[RES]] to %[[KWDEFAULTS]]
# CHECK: %[[CLOSURE:.*]] = py.getGlobalValue @builtins.None
# CHECK: py.setAttr "__closure__" of %[[RES]] to %[[CLOSURE]]
# CHECK: py.store %[[RES]] into %{{.*}}

# CHECK: func private @"foo.<locals>.bar$impl[0]"
# CHECK-SAME: %{{[[:alnum:]]+}}
# CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
# CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
# CHECK: %[[a:.*]] = py.alloc
# CHECK: py.store %[[ARG0]] into %[[a]]
# CHECK: %[[c:.*]] = py.alloc
# CHECK: py.store %[[ARG1]] into %[[c]]

# CHECK: func private @"foo.<locals>.bar$cc[0]"
# CHECK-SAME: %[[SELF:[[:alnum:]]+]]
# CHECK-SAME: %[[TUPLE:[[:alnum:]]+]]
# CHECK-SAME: %[[DICT:[[:alnum:]]+]]
# CHECK: %[[DEFAULT_TUPLE:.*]], %{{.*}} = py.getAttr "__defaults__" from %[[SELF]]
# CHECK: %[[DEFAULT_DICT:.*]], %{{.*}} = py.getAttr "__kwdefaults__" from %[[SELF]]
# CHECK: %[[TUPLE_LEN:.*]] = py.tuple.integer.len %[[TUPLE]]

# CHECK: %[[INDEX:.*]] = constant 0
# CHECK: %[[IS_LESS:.*]] = cmpi ult, %[[INDEX]], %[[TUPLE_LEN]]
# CHECK: cond_br %[[IS_LESS]], ^[[LESS_BLOCK:.*]], ^[[UNBOUND_BLOCK:[[:alnum:]]+]]

# CHECK: ^[[UNBOUND_BLOCK]]:
# CHECK: %[[UNBOUND:.*]] = py.unboundValue
# CHECK: br ^[[RESULT_BLOCK:[[:alnum:]]+]](
# CHECK-SAME: %[[UNBOUND]]
# CHECK-SAME: )

# CHECK: ^[[LESS_BLOCK]]:
# CHECK: %[[VALUE:.*]] = py.tuple.integer.getItem %[[TUPLE]][
# CHECK-SAME: %[[INDEX]]
# CHECK-SAME: ]
# CHECK: br ^[[RESULT_BLOCK]](
# CHECK-SAME: %[[VALUE]]
# CHECK-SAME: )

# CHECK: ^[[RESULT_BLOCK]](
# CHECK-SAME: %[[ARG_VALUE:[[:alnum:]]+]]
# CHECK-SAME: )

# CHECK: %[[INDEX:.*]] = py.constant "a"
# CHECK: %[[VALUE:.*]], %[[FOUND:.*]] = py.dict.tryGetItem %[[DICT]][%[[INDEX]]]
# CHECK: cond_br %[[FOUND]], ^[[FOUND_BLOCK:.*]], ^[[NOT_FOUND_BLOCK:[[:alnum:]]+]]

# CHECK: ^[[NOT_FOUND_BLOCK]]:
# CHECK: %[[UNBOUND:.*]] = py.unboundValue
# CHECK: br ^[[RESULT_BLOCK:[[:alnum:]]+]](
# CHECK-SAME: %[[UNBOUND]]
# CHECK-SAME: )

# CHECK: ^[[FOUND_BLOCK]]:
# CHECK: %[[IS_UNBOUND:.*]] = py.isUnboundValue %[[ARG_VALUE]]
# CHECK: cond_br %[[IS_UNBOUND]], ^[[RESULT_BLOCK]](
# CHECK-SAME: %[[VALUE]]
# CHECK-SAME: )
# CHECK-SAME: ^[[BOUND_BLOCK:[[:alnum:]]+]]

# CHECK: ^[[BOUND_BLOCK]]:
# exception creation code...
# CHECK: py.raise %{{[[:alnum:]]+}}

# CHECK: ^[[RESULT_BLOCK]](
# CHECK-SAME: %[[ARG_VALUE:[[:alnum:]]+]]
# CHECK-SAME: ):

# CHECK: %[[IS_UNBOUND:.*]] = py.isUnboundValue %[[ARG_VALUE]]
# CHECK: cond_br %[[IS_UNBOUND]], ^[[UNBOUND_BLOCK:[[:alnum:]]+]], ^[[BOUND_BLOCK:[[:alnum:]]+]](
# CHECK-SAME: %[[ARG_VALUE]]
# CHECK-SAME: )

# CHECK: ^[[UNBOUND_BLOCK]]:
# CHECK: %[[INDEX:.*]] = constant 0
# CHECK: %[[DEFAULT_ARG:.*]] = py.tuple.integer.getItem %[[DEFAULT_TUPLE]][
# CHECK-SAME: %[[INDEX]]
# CHECK-SAME: ]
# CHECK: br ^[[BOUND_BLOCK]](
# CHECK-SAME: %[[DEFAULT_ARG]]
# CHECK-SAME: )

# CHECK: ^[[BOUND_BLOCK]](
# CHECK-SAME: %{{[[:alnum:]]+}}
# CHECK-SAME: )

# CHECK-NOT: constant 1

# CHECK: %[[INDEX:.*]] = py.constant "c"
# CHECK: %[[VALUE:.*]], %[[FOUND:.*]] = py.dict.tryGetItem %[[DICT]][%[[INDEX]]]
# CHECK: cond_br %[[FOUND]], ^[[FOUND_BLOCK:.*]], ^[[NOT_FOUND_BLOCK:[[:alnum:]]+]]

# CHECK: ^[[NOT_FOUND_BLOCK]]:
# CHECK: %[[UNBOUND:.*]] = py.unboundValue
# CHECK: br ^[[RESULT_BLOCK:[[:alnum:]]+]](
# CHECK-SAME: %[[UNBOUND]]
# CHECK-SAME: )

# CHECK: ^[[FOUND_BLOCK]]:
# CHECK-NOT: py.isUnboundValue
# CHECK: br ^[[RESULT_BLOCK]](
# CHECK-SAME: %[[VALUE]]
# CHECK-SAME: )

# CHECK-NOT: py.raise

# CHECK: ^[[RESULT_BLOCK]](
# CHECK-SAME: %[[ARG_VALUE:[[:alnum:]]+]]
# CHECK-SAME: ):

# CHECK: %[[IS_UNBOUND:.*]] = py.isUnboundValue %[[ARG_VALUE]]
# CHECK: cond_br %[[IS_UNBOUND]], ^[[UNBOUND_BLOCK:[[:alnum:]]+]], ^[[BOUND_BLOCK:[[:alnum:]]+]](
# CHECK-SAME: %[[ARG_VALUE]]
# CHECK-SAME: )

# CHECK: ^[[UNBOUND_BLOCK]]:
# CHECK: %[[INDEX:.*]] = py.constant "c"
# CHECK: %[[DEFAULT_ARG:.*]], %{{.*}} = py.dict.tryGetItem %[[DEFAULT_DICT]][%[[INDEX]]]
# CHECK: br ^[[BOUND_BLOCK]](
# CHECK-SAME: %[[DEFAULT_ARG]]
# CHECK-SAME: )

# CHECK: ^[[BOUND_BLOCK]](
# CHECK-SAME: %{{[[:alnum:]]+}}
# CHECK-SAME: )
