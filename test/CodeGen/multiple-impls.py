# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s


def foo():
    global x
    x = 3


def foo():
    global x
    x = 2

# CHECK: %[[RES:.*]] = py.makeFunc @"foo$cc[0]"
# CHECK: py.store %[[RES]] : !py.dynamic into @foo

# CHECK: %[[RES:.*]] = py.makeFunc @"foo$cc[1]"
# CHECK: py.store %[[RES]] : !py.dynamic into @foo

# CHECK-DAG: func private @"foo$impl[0]"
# CHECK-DAG: func private @"foo$impl[1]"
