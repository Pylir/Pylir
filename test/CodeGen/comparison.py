# RUN: pylir %s -emit-mlir -o - | FileCheck %s

# XFAIL: *

1 < 3 < 5

# CHECK-LABEL: __init__
# CHECK: %[[ONE:.*]] = py.constant #py.int<1>
# CHECK: %[[THREE:.*]] = py.constant #py.int<3>
# CHECK: %[[LESS:.*]] = py.less %[[ONE]], %[[THREE]]
# CHECK: %[[LESS_BOOL:.*]] = py.bool %[[LESS]]
# CHECK: %[[LESS_I1:.*]] = py.bool.toI1 %[[LESS_BOOL]]
# CHECK: cond_br %[[LESS_I1]], ^[[FOUND:.*]](%[[LESS]] : !py.dynamic), ^[[NOT_FOUND:[[:alnum:]]+]]

# CHECK: ^[[NOT_FOUND]]:
# CHECK: %[[FIVE:.*]] = py.constant #py.int<5>
# CHECK: %[[LESS:.*]] = py.less %[[THREE]], %[[FIVE]]
# CHECK: br ^[[FOUND]](%[[LESS]] : !py.dynamic)

# CHECK: ^[[FOUND]]
# CHECK-SAME: %{{[[:alnum:]]+}}
