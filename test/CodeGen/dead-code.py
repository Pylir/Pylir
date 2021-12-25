# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

def foo():
    return
    1 + 2

# CHECK: func private @"foo$impl[0]"
# CHECK-NEXT: %[[NONE:.*]] = py.constant @builtins.None
# CHECK-NEXT: return %[[NONE]]
