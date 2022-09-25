# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

import pylir.intr.object


def foo():
    pass

t = pylir.intr.typeOf(foo)
print(pylir.intr.getSlot(foo, t, "__qualname__"))

# CHECK: %[[PRINT:.*]] = py.constant(#py.ref<@builtins.print>)
# CHECK: %[[FOO:.*]] = py.load @foo$handle
# CHECK: %[[NAME:.*]] = py.getSlot "__qualname__" from %[[FOO]]
# CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[NAME]])
# CHECK: py.call @pylir__call__(%[[PRINT]], %[[TUPLE]], %{{.*}})

pylir.intr.int.cmp("ne", 0, 1)

# CHECK: %[[ZERO:.*]] = py.constant(#py.int<0>)
# CHECK: %[[ONE:.*]] = py.constant(#py.int<1>)
# CHECK: %[[CMP:.*]] = py.int.cmp ne %[[ZERO]], %[[ONE]]
# CHECK: %[[BOOL:.*]] = py.bool.fromI1 %[[CMP]]

pylir.intr.function.call(t[0], t[0], (5,), {})

# CHECK: %[[T:.*]] = py.load @t$handle
# CHECK: %[[ZERO:.*]] = py.constant(#py.int<0>)
# CHECK: %[[ITEM1:.*]] = py.call @pylir__getitem__(%[[T]], %[[ZERO]])
# CHECK: %[[T:.*]] = py.load @t$handle
# CHECK: %[[ZERO:.*]] = py.constant(#py.int<0>)
# CHECK: %[[ITEM2:.*]] = py.call @pylir__getitem__(%[[T]], %[[ZERO]])
# CHECK: %[[FIVE:.*]] = py.constant(#py.int<5>)
# CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[FIVE:.*]])
# CHECK: %[[DICT:.*]] = py.makeDict ()
# CHECK: py.function.call %[[ITEM1]](%[[ITEM2]], %[[TUPLE]], %[[DICT]])
