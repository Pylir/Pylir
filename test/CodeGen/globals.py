# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK-DAG: globalHandle "private" @x
# CHECK-DAG: globalHandle "private" @y
# CHECK-DAG: globalHandle "private" @z
# CHECK-DAG: globalHandle "private" @foo

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
