# RUN: pylir %s -emit-mlir -o - | FileCheck %s

def foo():
    return 3

# CHECK-LABEL: @"foo$impl[0]"
# CHECK: %[[THREE:.*]] = py.constant #py.int<3>
# CHECK: return %[[THREE]]
