# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK-DAG: global "private" @x
# CHECK-DAG: global "private" @y
# CHECK-DAG: global "private" @z
# CHECK-DAG: global "private" @foo

# CHECK-LABEL: @__init__
# CHECK: %[[UNBOUND:.*]] = constant(#py.unbound)
# CHECK-DAG: store %[[UNBOUND]] : !py.dynamic into @x
# CHECK-DAG: store %[[UNBOUND]] : !py.dynamic into @y
# CHECK-DAG: store %[[UNBOUND]] : !py.dynamic into @z
# CHECK-DAG: store %[[UNBOUND]] : !py.dynamic into @foo

x = 2

# CHECK-DAG: %[[VALUE:.*]] = constant(#py.int<2>)
# CHECK: store %[[VALUE]] : !py.dynamic into @x

x


# CHECK: load @x

def foo():
    global y
    y = 3


(z := 3)

# CHECK-DAG: %[[VALUE:.*]] = constant(#py.int<3>)
# CHECK: store %[[VALUE]] : !py.dynamic into @z

# CHECK-LABEL: func private @"foo$impl[0]"
# CHECK: %[[VALUE:.*]] = constant(#py.int<3>)
# CHECK: store %[[VALUE]] : !py.dynamic into @y
