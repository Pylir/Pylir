# RUN: pylir %s -emit-mlir -o - | FileCheck %s

# CHECK-LABEL: @__init__

()

# CHECK: makeTuple
# CHECK-SAME: ()

(7,)

# CHECK: %[[OP1:.*]] = py.constant #py.int<7>
# CHECK: makeTuple
# CHECK-SAME: %[[OP1]]

(5, 3)

# CHECK: %[[OP1:.*]] = py.constant #py.int<5>
# CHECK: %[[OP2:.*]] = py.constant #py.int<3>
# CHECK: makeTuple
# CHECK-SAME: %[[OP1]], %[[OP2]]

(*(), 3)

# CHECK: %[[OP1:.*]] = py.makeTuple ()
# CHECK: %[[OP2:.*]] = py.constant #py.int<3>
# CHECK: makeTuple
# CHECK-SAME: *%[[OP1]], %[[OP2]]
