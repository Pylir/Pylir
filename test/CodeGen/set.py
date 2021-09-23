# RUN: pylir %s -emit-mlir -o - | FileCheck %s

# CHECK-LABEL: @__init__

{7}

# CHECK: %[[OP1:.*]] = py.constant #py.int<7>
# CHECK: makeSet
# CHECK-SAME: %[[OP1]]

{5, 3}

# CHECK: %[[OP1:.*]] = py.constant #py.int<5>
# CHECK: %[[OP2:.*]] = py.constant #py.int<3>
# CHECK: makeSet
# CHECK-SAME: %[[OP1]], %[[OP2]]

{*(), 3}

# CHECK: %[[OP1:.*]] = py.makeTuple ()
# CHECK: %[[OP2:.*]] = py.constant #py.int<3>
# CHECK: makeSet
# CHECK-SAME: *%[[OP1]], %[[OP2]]