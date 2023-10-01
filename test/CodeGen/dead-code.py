# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK: #[[NONE_ATTR:.*]] = #py.globalValue<builtins.None,
def foo():
    return
    1 + 2

# CHECK: func private @"foo$impl[0]"
# CHECK: %[[NONE:.*]] = constant(#[[NONE_ATTR]])
# CHECK-NEXT: return %[[NONE]]
