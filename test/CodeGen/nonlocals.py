# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

def foo():
    x = 3

    def bar():
        return x

# CHECK-LABEL: func private @"foo$impl[0]"
# CHECK: %[[CELL:.*]] = py.constant(#py.ref<@builtins.cell>)
# CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[CELL]])
# CHECK: %[[NEW_METHOD:.*]] = py.getSlot %[[CELL]][%{{.*}}]
# CHECK: %[[X:.*]] = py.function.call %[[NEW_METHOD]](%[[NEW_METHOD]], %[[TUPLE]], %{{[[:alnum:]]+}})

# CHECK: %[[BAR:.*]] = py.makeFunc @"foo.<locals>.bar$cc[0]"
# CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[X]])
# CHECK: py.setSlot %[[BAR]][%{{.*}}] to %[[TUPLE]]


# CHECK-LABEL: func private @"foo.<locals>.bar$impl[0]"
# CHECK-SAME: %[[SELF:[[:alnum:]]+]]
# CHECK: %[[TUPLE:.*]] = py.getSlot %[[SELF]][%{{.*}}]
# CHECK: %[[ZERO:.*]] = arith.constant 0 : index
# CHECK: %[[X:.*]] = py.tuple.getItem %[[TUPLE]][%[[ZERO]]]
# CHECK: %[[VALUE:.*]] = py.getSlot %[[X]][%{{.*}}]
# CHECK: return %[[VALUE]]
