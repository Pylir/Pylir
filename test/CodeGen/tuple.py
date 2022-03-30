# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK-LABEL: @__init__

()

# CHECK: py.constant(#py.tuple<value = ()>)

(7,)

# CHECK: %[[OP1:.*]] = py.constant(#py.int<value = 7>)
# CHECK: makeTuple
# CHECK-SAME: %[[OP1]]

(5, 3)

# CHECK: %[[OP1:.*]] = py.constant(#py.int<value = 5>)
# CHECK: %[[OP2:.*]] = py.constant(#py.int<value = 3>)
# CHECK: makeTuple
# CHECK-SAME: %[[OP1]], %[[OP2]]

(*(), 3)

# CHECK: %[[OP1:.*]] = py.constant(#py.tuple<value = ()>)
# CHECK: %[[OP2:.*]] = py.constant(#py.int<value = 3>)
# CHECK: makeTuple
# CHECK-SAME: *%[[OP1]], %[[OP2]]
