# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

def foo():
    return
    1 + 2

# CHECK: func private @"foo$impl[0]"
# CHECK: %[[NONE:.*]] = constant(#py.ref<@builtins.None>)
# CHECK-NEXT: return %[[NONE]]
