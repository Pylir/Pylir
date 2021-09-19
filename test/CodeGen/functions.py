# RUN: pylir %s -emit-mlir -o - | FileCheck %s

# CHECK: @foo

# CHECK-LABEL: __init__

# CHECK: %[[RES:.*]] = py.makeFunc @"foo$impl[0]"
# CHECK: %[[FOO:.*]] = py.getGlobal @foo
# CHECK: py.store %[[RES]] into %[[FOO]]

def foo():
    x = 3

# CHECK: func private @"foo$impl[0]"

# CHECK: %[[X:.*]] = py.alloca
# CHECK: %[[THREE:.*]] = py.constant #py.int<3>
# CHECK: py.store %[[THREE]] into %[[X]]


