# RUN: pylir %s -emit-mlir -o - | FileCheck %s

# CHECK-LABEL: @__init__

(5,)[0]

# CHECK: %[[OP1:.*]] = py.makeTuple
# CHECK: %[[OP2:.*]] = py.constant #py.int<0>
# CHECK: getItem %[[OP1]][%[[OP2]]]
