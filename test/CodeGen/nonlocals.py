# RUN: pylir %s -emit-mlir -o - | FileCheck %s

def foo():
    x = 3

    def bar():
        return x

# CHECK-LABEL: func private @"foo$impl[0]"
# CHECK: %[[CELL:.*]] = py.singleton cell
# CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[CELL]])
# CHECK: %[[NEW_METHOD:.*]], %{{[[:alnum:]]+}} = py.getAttr "__new__" from %[[CELL]]
# CHECK: %[[CALLABLE:.*]] = py.function.getFunction %[[NEW_METHOD]]
# CHECK: %[[X:.*]] = call_indirect %[[CALLABLE]](%[[NEW_METHOD]], %[[TUPLE]], %{{[[:alnum:]]+}})

# CHECK: %[[BAR:.*]] = py.makeFunc @"foo.<locals>.bar$cc[0]"
# CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[X]])
# CHECK: py.setAttr "__closure__" of %[[BAR]] to %[[TUPLE]]


# CHECK-LABEL: func private @"foo.<locals>.bar$impl[0]"
# CHECK-SAME: %[[SELF:[[:alnum:]]+]]
# CHECK: %[[TUPLE:.*]], %{{.*}} = py.getAttr "__closure__" from %[[SELF]]
# CHECK: %[[ZERO:.*]] = constant 0 : index
# CHECK: %[[X:.*]] = py.tuple.integer.getItem %[[TUPLE]][%[[ZERO]] : index]
# CHECK: %[[VALUE:.*]], %{{.*}} = py.getAttr "cell_contents" from %[[X]]
# CHECK: return %[[VALUE]]
