# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK-LABEL: init "__main__"

()

# CHECK: makeTuple ()

(7,)

# CHECK: %[[OP1:.*]] = py.constant(#py.int<7>)
# CHECK: makeTuple
# CHECK-SAME: %[[OP1]]

(5, 3)

# CHECK: %[[OP1:.*]] = py.constant(#py.int<5>)
# CHECK: %[[OP2:.*]] = py.constant(#py.int<3>)
# CHECK: makeTuple
# CHECK-SAME: %[[OP1]], %[[OP2]]

(*(), 3)

# CHECK: %[[OP1:.*]] = py.makeTuple ()
# CHECK: %[[OP2:.*]] = py.constant(#py.int<3>)
# CHECK: makeTuple
# CHECK-SAME: *%[[OP1]], %[[OP2]]
