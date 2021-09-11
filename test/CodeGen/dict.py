# RUN: pylir %s -emit-mlir -o - | FileCheck %s

# CHECK-LABEL: __init__

{}

# CHECK: makeDict
# CHECK-SAME: ()

{7: 5}

# CHECK: %[[OP1:.*]] = py.constant #py.int<7>
# CHECK: %[[OP2:.*]] = py.constant #py.int<5>
# CHECK: makeDict
# CHECK-SAME: %[[OP1]] : %[[OP2]]

{7: 5, 5: 3}

# CHECK: %[[OP1:.*]] = py.constant #py.int<7>
# CHECK: %[[OP2:.*]] = py.constant #py.int<5>
# CHECK: %[[OP3:.*]] = py.constant #py.int<5>
# CHECK: %[[OP4:.*]] = py.constant #py.int<3>
# CHECK: makeDict
# CHECK-SAME: %[[OP1]] : %[[OP2]], %[[OP3]] : %[[OP4]]

{**{}, 3: 5}

# CHECK: %[[OP1:.*]] = py.makeDict ()
# CHECK: %[[OP2:.*]] = py.constant #py.int<3>
# CHECK: %[[OP3:.*]] = py.constant #py.int<5>
# CHECK: makeDict
# CHECK-SAME: **%[[OP1]], %[[OP2]] : %[[OP3]]
