# RUN: pylir %s -emit-mlir -o - | FileCheck %s

x = 2


def foo():
    global y


(z := 3)

# CHECK: global @x
# CHECK: global @y
# CHECK: global @z
