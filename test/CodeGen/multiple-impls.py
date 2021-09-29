# RUN: pylir %s -emit-mlir -o - | FileCheck %s


def foo():
    global x
    x = 3


def foo():
    global x
    x = 2

# CHECK: %[[RES:.*]] = py.makeFunc @"foo$impl[0]"
# CHECK: %[[FOO:.*]] = py.getGlobalHandle @foo
# CHECK: py.store %[[RES]] into %[[FOO]]

# CHECK: %[[RES:.*]] = py.makeFunc @"foo$impl[1]"
# CHECK: %[[FOO:.*]] = py.getGlobalHandle @foo
# CHECK: py.store %[[RES]] into %[[FOO]]

# CHECK-DAG: func private @"foo$impl[0]"
# CHECK-DAG: func private @"foo$impl[1]"
