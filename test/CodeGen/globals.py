# RUN: pylir %s -emit-pylir -o - | FileCheck %s

# CHECK-DAG: globalHandle @x
# CHECK-DAG: globalHandle @y
# CHECK-DAG: globalHandle @z
# CHECK-DAG: globalHandle @foo

x = 2

# CHECK-DAG: %[[VALUE:.*]] = py.constant #py.int<2>
# CHECK: py.store %[[VALUE]] into @x

x


# CHECK: py.load @x

def foo():
    global y
    y = 3


(z := 3)

# CHECK-DAG: %[[VALUE:.*]] = py.constant #py.int<3>
# CHECK: py.store %[[VALUE]] into @z

# CHECK-LABEL: func private @"foo$impl[0]"
# CHECK: %[[VALUE:.*]] = py.constant #py.int<3>
# CHECK: py.store %[[VALUE]] into @y
