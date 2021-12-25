# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK-LABEL: __init__

# CHECK: %[[RES:.*]] = py.makeFunc @"foo$cc[0]"
# CHECK: %[[NAME:.*]] = py.constant #py.str<"foo">
# CHECK: py.setSlot "__name__" of %[[RES]] : %{{.*}} to %[[NAME]]
# CHECK: %[[NAME:.*]] = py.constant #py.str<"foo">
# CHECK: py.setSlot "__qualname__" of %[[RES]] : %{{.*}} to %[[NAME]]
# CHECK: %[[DEFAULTS:.*]] = py.constant @builtins.None
# CHECK: py.setSlot "__defaults__" of %[[RES]] : %{{.*}} to %[[DEFAULTS]]
# CHECK: %[[KWDEFAULTS:.*]] = py.constant @builtins.None
# CHECK: py.setSlot "__kwdefaults__" of %[[RES]] : %{{.*}} to %[[KWDEFAULTS]]
# CHECK: %[[CLOSURE:.*]] = py.constant @builtins.None
# CHECK: py.setSlot "__closure__" of %[[RES]] : %{{.*}} to %[[CLOSURE]]
# CHECK: py.store %[[RES]] into @foo

def foo():
    x = 3

    def bar(a=3, *, c=1):
        pass

# CHECK-LABEL: func private @"foo$impl[0]"

# CHECK: %[[THREE:.*]] = py.constant #py.int<3>
# CHECK: %[[THREE:.*]] = py.constant #py.int<3>
# CHECK: %[[ONE:.*]] = py.constant #py.int<1>
# CHECK: %[[C:.*]] = py.constant #py.str<"c">
# CHECK: %[[RES:.*]] = py.makeFunc @"foo.<locals>.bar$cc[0]"
# CHECK: %[[NAME:.*]] = py.constant #py.str<"bar">
# CHECK: py.setSlot "__name__" of %[[RES]] : %{{.*}} to %[[NAME]]
# CHECK: %[[NAME:.*]] = py.constant #py.str<"foo.<locals>.bar">
# CHECK: py.setSlot "__qualname__" of %[[RES]] : %{{.*}} to %[[NAME]]
# CHECK: %[[DEFAULTS:.*]] = py.makeTuple (%[[THREE]])
# CHECK: py.setSlot "__defaults__" of %[[RES]] : %{{.*}} to %[[DEFAULTS]]
# CHECK: %[[KWDEFAULTS:.*]] = py.makeDict (%[[C]] : %[[ONE]])
# CHECK: py.setSlot "__kwdefaults__" of %[[RES]] : %{{.*}} to %[[KWDEFAULTS]]
# CHECK: %[[CLOSURE:.*]] = py.constant @builtins.None
# CHECK: py.setSlot "__closure__" of %[[RES]] : %{{.*}} to %[[CLOSURE]]

# CHECK: func private @"foo.<locals>.bar$impl[0]"
# CHECK-SAME: %{{[[:alnum:]]+}}
# CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
# CHECK-SAME: %[[ARG1:[[:alnum:]]+]]

# CHECK: func private @"foo.<locals>.bar$cc[0]"
# CHECK-SAME: %[[SELF:[[:alnum:]]+]]
# CHECK-SAME: %[[TUPLE:[[:alnum:]]+]]
# CHECK-SAME: %[[DICT:[[:alnum:]]+]]
# CHECK: %[[DEFAULT_TUPLE:.*]] = py.getSlot "__defaults__" from %[[SELF]]
# CHECK: %[[DEFAULT_DICT:.*]] = py.getSlot "__kwdefaults__" from %[[SELF]]
# CHECK: %[[TUPLE_LEN:.*]] = py.tuple.len %[[TUPLE]]

# CHECK: %[[INDEX:.*]] = arith.constant 0
# CHECK: %[[IS_LESS:.*]] = arith.cmpi ult, %[[INDEX]], %[[TUPLE_LEN]]
# CHECK: cond_br %[[IS_LESS]], ^[[LESS_BLOCK:.*]], ^[[UNBOUND_BLOCK:[[:alnum:]]+]]

# CHECK: ^[[UNBOUND_BLOCK]]:
# CHECK: %[[UNBOUND:.*]] = py.constant #py.unbound
# CHECK: br ^[[RESULT_BLOCK:[[:alnum:]]+]](
# CHECK-SAME: %[[UNBOUND]]
# CHECK-SAME: )

# CHECK: ^[[LESS_BLOCK]]:
# CHECK: %[[VALUE:.*]] = py.tuple.getItem %[[TUPLE]][
# CHECK-SAME: %[[INDEX]]
# CHECK-SAME: ]
# CHECK: br ^[[RESULT_BLOCK]](
# CHECK-SAME: %[[VALUE]]
# CHECK-SAME: )

# CHECK: ^[[RESULT_BLOCK]](
# CHECK-SAME: %[[ARG_VALUE:[[:alnum:]]+]]
# CHECK-SAME: )

# CHECK: %[[INDEX:.*]] = py.constant #py.str<"a">
# CHECK: %[[VALUE:.*]], %[[FOUND:.*]] = py.dict.tryGetItem %[[DICT]][%[[INDEX]]]
# CHECK: cond_br %[[FOUND]], ^[[FOUND_BLOCK:.*]], ^[[NOT_FOUND_BLOCK:[[:alnum:]]+]]

# CHECK: ^[[NOT_FOUND_BLOCK]]:
# CHECK: %[[UNBOUND:.*]] = py.constant #py.unbound
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
# CHECK: %[[INDEX:.*]] = arith.constant 0
# CHECK: %[[DEFAULT_ARG:.*]] = py.tuple.getItem %[[DEFAULT_TUPLE]][
# CHECK-SAME: %[[INDEX]]
# CHECK-SAME: ]
# CHECK: br ^[[BOUND_BLOCK]](
# CHECK-SAME: %[[DEFAULT_ARG]]
# CHECK-SAME: )

# CHECK: ^[[BOUND_BLOCK]](
# CHECK-SAME: %{{[[:alnum:]]+}}
# CHECK-SAME: )

# CHECK-NOT: arith.constant 1

# CHECK: %[[INDEX:.*]] = py.constant #py.str<"c">
# CHECK: %[[VALUE:.*]], %[[FOUND:.*]] = py.dict.tryGetItem %[[DICT]][%[[INDEX]]]
# CHECK: cond_br %[[FOUND]], ^[[FOUND_BLOCK:.*]], ^[[NOT_FOUND_BLOCK:[[:alnum:]]+]]

# CHECK: ^[[NOT_FOUND_BLOCK]]:
# CHECK: %[[UNBOUND:.*]] = py.constant #py.unbound
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
# CHECK: %[[INDEX:.*]] = py.constant #py.str<"c">
# CHECK: %[[DEFAULT_ARG:.*]], %{{.*}} = py.dict.tryGetItem %[[DEFAULT_DICT]][%[[INDEX]]]
# CHECK: br ^[[BOUND_BLOCK]](
# CHECK-SAME: %[[DEFAULT_ARG]]
# CHECK-SAME: )

# CHECK: ^[[BOUND_BLOCK]](
# CHECK-SAME: %{{[[:alnum:]]+}}
# CHECK-SAME: )
