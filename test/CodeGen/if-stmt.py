# RUN: pylir %s -emit-mlir -o - | FileCheck %s

# CHECK-LABEL: __init__

if 3:
    5

# CHECK: %[[THREE:.*]] = py.constant #py.int<3>
# CHECK: %[[THREE_BOOL:.*]] = py.bool %[[THREE]]
# CHECK: %[[THREE_I1:.*]] = py.bool.toI1 %[[THREE_BOOL]]
# CHECK: cond_br %[[THREE_I1]], ^[[TRUE:.*]], ^[[THEN:[[:alnum:]]+]]

# CHECK: ^[[TRUE]]:
# CHECK: py.constant #py.int<5>
# CHECK: br ^[[THEN]]

# CHECK: ^[[THEN]]:

if 3:
    5
else:
    4

# CHECK: %[[THREE:.*]] = py.constant #py.int<3>
# CHECK: %[[THREE_BOOL:.*]] = py.bool %[[THREE]]
# CHECK: %[[THREE_I1:.*]] = py.bool.toI1 %[[THREE_BOOL]]
# CHECK: cond_br %[[THREE_I1]], ^[[TRUE:.*]], ^[[ELSE:[[:alnum:]]+]]

# CHECK: ^[[TRUE]]:
# CHECK: py.constant #py.int<5>
# CHECK: br ^[[THEN:[[:alnum:]]+]]

# CHECK: ^[[ELSE]]:
# CHECK: py.constant #py.int<4>
# CHECK: br ^[[THEN]]

# CHECK: ^[[THEN]]:


if 3:
    5
elif 4:
    6

# CHECK: %[[THREE:.*]] = py.constant #py.int<3>
# CHECK: %[[THREE_BOOL:.*]] = py.bool %[[THREE]]
# CHECK: %[[THREE_I1:.*]] = py.bool.toI1 %[[THREE_BOOL]]
# CHECK: cond_br %[[THREE_I1]], ^[[TRUE:.*]], ^[[ELIF:[[:alnum:]]+]]

# CHECK: ^[[TRUE]]:
# CHECK: py.constant #py.int<5>
# CHECK: br ^[[THEN:[[:alnum:]]+]]

# CHECK: ^[[ELIF]]:
# CHECK: %[[FOUR:.*]] = py.constant #py.int<4>
# CHECK: %[[FOUR_BOOL:.*]] = py.bool %[[FOUR]]
# CHECK: %[[FOUR_I1:.*]] = py.bool.toI1 %[[FOUR_BOOL]]
# CHECK: cond_br %[[FOUR_I1]], ^[[ELIF_TRUE:.*]], ^[[THEN]]

# CHECK: ^[[ELIF_TRUE]]:
# CHECK: py.constant #py.int<6>
# CHECK: br ^[[THEN]]

# CHECK: ^[[THEN]]:


if 3:
    5
elif 4:
    6
else:
    7

# CHECK: %[[THREE:.*]] = py.constant #py.int<3>
# CHECK: %[[THREE_BOOL:.*]] = py.bool %[[THREE]]
# CHECK: %[[THREE_I1:.*]] = py.bool.toI1 %[[THREE_BOOL]]
# CHECK: cond_br %[[THREE_I1]], ^[[TRUE:.*]], ^[[ELIF:[[:alnum:]]+]]

# CHECK: ^[[TRUE]]:
# CHECK: py.constant #py.int<5>
# CHECK: br ^[[THEN:[[:alnum:]]+]]

# CHECK: ^[[ELIF]]:
# CHECK: %[[FOUR:.*]] = py.constant #py.int<4>
# CHECK: %[[FOUR_BOOL:.*]] = py.bool %[[FOUR]]
# CHECK: %[[FOUR_I1:.*]] = py.bool.toI1 %[[FOUR_BOOL]]
# CHECK: cond_br %[[FOUR_I1]], ^[[ELIF_TRUE:.*]], ^[[ELSE:[[:alnum:]]+]]

# CHECK: ^[[ELIF_TRUE]]:
# CHECK: py.constant #py.int<6>
# CHECK: br ^[[THEN]]

# CHECK: ^[[ELSE]]:
# CHECK: py.constant #py.int<7>
# CHECK: br ^[[THEN]]

# CHECK: ^[[THEN]]:
