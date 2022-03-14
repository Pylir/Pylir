# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK-DAG: globalHandle "private" @x
# CHECK-DAG: globalHandle "private" @y
# CHECK-DAG: globalHandle "private" @z
# CHECK-DAG: globalHandle "private" @foo

# CHECK-LABEL: @__init__
# CHECK: %[[UNBOUND:.*]] = py.constant #py.unbound
# CHECK-DAG: py.store %[[UNBOUND]] into @x
# CHECK-DAG: py.store %[[UNBOUND]] into @y
# CHECK-DAG: py.store %[[UNBOUND]] into @z
# CHECK-DAG: py.store %[[UNBOUND]] into @foo

x = 2

# CHECK-DAG: %[[VALUE:.*]] = py.constant #py.int<value = 2>
# CHECK: py.store %[[VALUE]] into @x

x


# CHECK: py.load @x

def foo():
    global y
    y = 3


(z := 3)

# CHECK-DAG: %[[VALUE:.*]] = py.constant #py.int<value = 3>
# CHECK: py.store %[[VALUE]] into @z

# CHECK-LABEL: func private @"foo$impl[0]"
# CHECK: %[[VALUE:.*]] = py.constant #py.int<value = 3>
# CHECK: py.store %[[VALUE]] into @y
