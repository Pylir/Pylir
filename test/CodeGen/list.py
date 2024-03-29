# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK-LABEL: @__init__

[]

# CHECK: makeList
# CHECK-SAME: ()

[7]

# CHECK: %[[OP1:.*]] = constant(#py.int<7>)
# CHECK: makeList
# CHECK-SAME: %[[OP1]]

[5, 3]

# CHECK: %[[OP1:.*]] = constant(#py.int<5>)
# CHECK: %[[OP2:.*]] = constant(#py.int<3>)
# CHECK: makeList
# CHECK-SAME: %[[OP1]], %[[OP2]]

[*(), 3]

# CHECK: %[[OP1:.*]] = makeTuple ()
# CHECK: %[[OP2:.*]] = constant(#py.int<3>)
# CHECK: makeList
# CHECK-SAME: *%[[OP1]], %[[OP2]]
