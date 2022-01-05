# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

def foo(x):
    y = 5
    if x:
        y = 3
    return y


# CHECK-LABEL: func private @"foo$impl[0]"
# CHECK: %[[C0:.*]] = py.constant #py.int<5>
# ...
# CHECK: %[[COND:.*]] = py.bool.toI1
# CHECK: cond_br %[[COND]], ^[[TRUE_BLOCK:.*]], ^[[MERGE_BLOCK:[[:alnum:]]+]]
# CHECK-SAME: %[[C0]]

# CHECK: ^[[TRUE_BLOCK]]:
# CHECK: %[[C1:.*]] = py.constant #py.int<3>
# CHECK: br ^[[MERGE_BLOCK]]
# CHECK-SAME: %[[C1]]

# CHECK: ^[[MERGE_BLOCK]]
# CHECK-SAME: %[[RESULT:[[:alnum:]]+]]
# CHECK: return %[[RESULT]]

def bar(x):
    while x:
        x = x()
    return x
