# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

def foo():
    return 3

# CHECK-LABEL: @"foo$impl[0]"
# CHECK: %[[THREE:.*]] = py.constant(#py.int<value = 3>)
# CHECK: return %[[THREE]]
