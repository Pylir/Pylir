# RUN: pylir %s -emit-mlir -o - | FileCheck %s

# CHECK-LABEL: __init__

object

# CHECK: py.getGlobalValue @builtins.object

# CHECK: %[[VALUE:.*]] = py.load @BaseException
# CHECK: py.store %[[VALUE]] into @x
x = BaseException

BaseException = True


def foo():
    TypeError = 5
    return TypeError

# CHECK-LABEL: func private @"foo$impl[0]"

# CHECK: %[[FIVE:.*]] = py.constant #py.int<5>
# CHECK: return %[[FIVE]]
