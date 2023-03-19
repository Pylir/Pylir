# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

import pylir.intr.object


def foo():
    pass

t = pylir.intr.typeOf(foo)

pylir.intr.int.cmp("ne", 0, 1)

# CHECK: %[[ZERO:.*]] = constant(#py.int<0>)
# CHECK: %[[ONE:.*]] = constant(#py.int<1>)
# CHECK: %[[CMP:.*]] = py.int.cmp ne %[[ZERO]], %[[ONE]]
# CHECK: %[[BOOL:.*]] = py.bool.fromI1 %[[CMP]]

pylir.intr.function.call(t[0], t[0], (5,), {})

# CHECK: %[[T:.*]] = load @t$handle
# CHECK: %[[ZERO:.*]] = constant(#py.int<0>)
# CHECK: %[[ITEM1:.*]] = call @pylir__getitem__(%[[T]], %[[ZERO]])
# CHECK: %[[T:.*]] = load @t$handle
# CHECK: %[[ZERO:.*]] = constant(#py.int<0>)
# CHECK: %[[ITEM2:.*]] = call @pylir__getitem__(%[[T]], %[[ZERO]])
# CHECK: %[[FIVE:.*]] = constant(#py.int<5>)
# CHECK: %[[TUPLE:.*]] = makeTuple (%[[FIVE:.*]])
# CHECK: %[[DICT:.*]] = makeDict ()
# CHECK: py.function.call %[[ITEM1]](%[[ITEM2]], %[[TUPLE]], %[[DICT]])
