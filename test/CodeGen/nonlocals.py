# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

def foo():
    x = 3

    def bar():
        return x

# CHECK-LABEL: func private @"foo$impl[0]"
# CHECK: %[[CELL:.*]] = constant(#py.ref<@builtins.cell>)
# CHECK: %[[TUPLE:.*]] = makeTuple (%[[CELL]])
# CHECK: %[[NEW_METHOD:.*]] = getSlot %[[CELL]][%{{.*}}]
# CHECK: %[[X:.*]] = function_call %[[NEW_METHOD]](%[[NEW_METHOD]], %[[TUPLE]], %{{[[:alnum:]]+}})

# CHECK: %[[BAR:.*]] = makeFunc @"foo.<locals>.bar$cc[0]"
# CHECK: %[[TUPLE:.*]] = makeTuple (%[[X]])
# CHECK: setSlot %[[BAR]][%{{.*}}] to %[[TUPLE]]


# CHECK-LABEL: func private @"foo.<locals>.bar$impl[0]"
# CHECK-SAME: %[[SELF:[[:alnum:]]+]]
# CHECK: %[[TUPLE:.*]] = getSlot %[[SELF]][%{{.*}}]
# CHECK: %[[ZERO:.*]] = arith.constant 0 : index
# CHECK: %[[X:.*]] = tuple_getItem %[[TUPLE]][%[[ZERO]]]
# CHECK: %[[VALUE:.*]] = getSlot %[[X]][%{{.*}}]
# CHECK: return %[[VALUE]]
