# RUN: pylir %s -emit-mlir -o - | FileCheck %s

def foo():
    return
    1 + 2

# CHECK: func private @"foo$impl[0]"
# CHECK-NEXT: %[[NONE:.*]] = py.getGlobalValue @builtins.None
# CHECK-NEXT: return %[[NONE]]
