# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

import pylir.intr.object


def foo():
    pass

t = pylir.intr.typeOf(foo)
print(pylir.intr.getSlot(foo, t, "__qualname__"))

# CHECK: %[[PRINT:.*]] = py.constant(@builtins.print)
# CHECK: %[[FOO:.*]] = py.load @foo
# CHECK: %[[NAME:.*]] = py.getSlot "__qualname__" from %[[FOO]]
# CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[NAME]])
# CHECK: py.call @pylir__call__(%[[PRINT]], %[[TUPLE]], %{{.*}})

pylir.intr.int.cmp("ne", 0, 1)

# CHECK: %[[ZERO:.*]] = py.constant(#py.int<0>)
# CHECK: %[[ONE:.*]] = py.constant(#py.int<1>)
# CHECK: %[[CMP:.*]] = py.int.cmp ne %[[ZERO]], %[[ONE]]
# CHECK: %[[BOOL:.*]] = py.bool.fromI1 %[[CMP]]
