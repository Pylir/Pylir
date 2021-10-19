# RUN: pylir %s -emit-mlir -o - | FileCheck %s

# CHECK-LABEL: __init__

object

# CHECK: py.getGlobalValue @builtins.object

# CHECK: %[[HANDLE:.*]] = py.getGlobalHandle @BaseException
# CHECK: %[[VALUE:.*]] = py.load %[[HANDLE]]
# CHECK: %[[X:.*]] = py.getGlobalHandle @x
# CHECK: py.store %[[VALUE]] into %[[X]]
x = BaseException

BaseException = True


def foo():
    TypeError = 5
    return TypeError

# CHECK-LABEL: func private @"foo$impl[0]"

# CHECK: %[[FIVE:.*]] = py.constant #py.int<5>
# CHECK: py.store %[[FIVE]] into %[[TYPE_ERROR:[[:alnum:]]+]]
# CHECK: %[[VALUE:.*]] = py.load %[[TYPE_ERROR]]
# CHECK: return %[[VALUE]]
