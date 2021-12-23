# RUN: pylir %s -emit-pylir -o - | FileCheck %s

def foo(x):
    y = 5
    if x:
        y = 3
    return y


# CHECK-LABEL: func private @"foo$impl[0]"
# CHECK: %[[C0:.*]] = py.constant #py.int<5>
# ...
# CHECK: %[[COND:.*]] = py.bool.toI1
# CHECK: cond_br %[[COND]], ^[[TRUE_BLOCK:.*]], ^[[FALSE_BLOCK:[[:alnum:]]+]]
# CHECK-SAME: %[[C0]]

# CHECK: ^[[TRUE_BLOCK]]:
# CHECK: %[[C1:.*]] = py.constant #py.int<3>
# CHECK: br ^[[FALSE_BLOCK]]
# CHECK-SAME: %[[C1]]

def bar(x):
    while x:
        x = x()
    return x

# CHECK-LABEL: func private @"bar$impl[0]"
# CHECK-SAME: %{{[[:alnum:]]+}}
# CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
# CHECK: br ^[[CONDITION:[[:alnum:]]+]]
# CHECK-SAME: %[[ARG0]]
# CHECK: ^[[CONDITION]]
# CHECK-SAME: // 2 preds:
# CHECK-SAME: ^{{[[:alnum:]]+}}
# CHECK-SAME: ^[[PRED:[[:alnum:]]+]]

# ...body

# CHECK: ^[[PRED]]:
# CHECK: %[[RESULT:.*]] = call_indirect
# CHECK: br ^[[CONDITION]]
# CHECK-SAME: %[[RESULT]]

