# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK: func "__main__.foo"
# CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
def foo(x):
    # CHECK: %[[CELL_TYPE:.*]] = py.constant
    # CHECK: %[[CELL:.*]] = call %[[CELL_TYPE]]()
    # CHECK: %[[ZERO:.*]] = arith.constant 0
    # CHECK: py.setSlot %[[CELL]][%[[ZERO]]] to %[[ARG0]]

    # CHECK: func "__main__.foo.<locals>.bar"
    # CHECK-SAME: %[[Y:[[:alnum:]]+]]
    # CHECK: %[[ZERO:.*]] = arith.constant 0
    # CHECK: %[[X:.*]] = py.getSlot %[[CELL]][%[[ZERO]]]
    # CHECK: %[[RES:.*]] = binAssignOp %[[X]] __iadd__ %[[Y]]
    # CHECK: %[[ZERO:.*]] = arith.constant 0
    # CHECK: py.setSlot %[[CELL]][%[[ZERO]]] to %[[RES]]
    def bar(y):
        nonlocal x
        x += y
