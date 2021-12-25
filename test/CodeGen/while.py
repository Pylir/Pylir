# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK-LABEL: @__init__

while True:
    pass

# CHECK: ^[[TEST:[[:alnum:]]+]]:
# CHECK: %[[COND:.*]] = py.constant #py.bool<True>
# CHECK-DAG: %[[DICT:.*]] = py.constant #py.dict<{}>
# CHECK-DAG: %[[TUPLE:.*]] = py.makeTuple (%[[COND]])
# CHECK: %[[COND_BOOL:.*]] = call_indirect %{{.*}}(%{{.*}}, %[[TUPLE]], %[[DICT]])
# CHECK: %[[COND_I1:.*]] = py.bool.toI1 %[[COND_BOOL]]
# CHECK: cond_br %[[COND_I1]], ^[[BODY:[[:alnum:]]+]], ^[[THEN:[[:alnum:]]+]]

# CHECK: ^[[BODY]]:
# CHECK: br ^[[TEST]]

# CHECK: ^[[THEN]]:

x = 0
while True:
    if x:
        break
else:
    pass

# CHECK: ^[[TEST:[[:alnum:]]+]]:
# __bool__ call...
# CHECK: %[[COND_I1:.*]] = py.bool.toI1
# CHECK: cond_br %[[COND_I1]], ^[[BODY:[[:alnum:]]+]], ^[[ELSE:[[:alnum:]]+]]

# CHECK: ^[[BODY]]:
# CHECK: py.load @x
# ... rest of if
# CHECK: br ^[[BREAK:[[:alnum:]]+]]

# CHECK: br ^[[TEST]]

# CHECK: ^[[ELSE]]:
# CHECK: br ^[[BREAK]]

# CHECK: ^[[BREAK]]:

while True:
    if x:
        continue
    else:
        break

# CHECK: ^[[TEST:[[:alnum:]]+]]:
# __bool__ call...
# CHECK: %[[COND_I1:.*]] = py.bool.toI1
# CHECK: cond_br %[[COND_I1]], ^[[BODY:[[:alnum:]]+]], ^[[THEN:[[:alnum:]]+]]

# CHECK: ^[[BODY]]:
# CHECK: py.load @x
# ... rest of if
# CHECK: br ^[[TEST]]

# CHECK: br ^[[THEN]]

# CHECK: ^[[THEN]]:
