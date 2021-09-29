# RUN: pylir %s -emit-mlir -o - | FileCheck %s

while True:
    pass

# CHECK: ^[[TEST:[[:alnum:]]+]]:
# CHECK: %[[COND:.*]] = py.constant #py.bool<True>
# CHECK: %[[COND_BOOL:.*]] = py.bool %[[COND]]
# CHECK: %[[COND_I1:.*]] = py.boolToI1 %[[COND_BOOL]]
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
# CHECK: %[[COND:.*]] = py.constant #py.bool<True>
# CHECK: %[[COND_BOOL:.*]] = py.bool %[[COND]]
# CHECK: %[[COND_I1:.*]] = py.boolToI1 %[[COND_BOOL]]
# CHECK: cond_br %[[COND_I1]], ^[[BODY:[[:alnum:]]+]], ^[[ELSE:[[:alnum:]]+]]

# CHECK: ^[[BODY]]:
# CHECK: py.getGlobalHandle @x
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
# CHECK: %[[COND:.*]] = py.constant #py.bool<True>
# CHECK: %[[COND_BOOL:.*]] = py.bool %[[COND]]
# CHECK: %[[COND_I1:.*]] = py.boolToI1 %[[COND_BOOL]]
# CHECK: cond_br %[[COND_I1]], ^[[BODY:[[:alnum:]]+]], ^[[THEN:[[:alnum:]]+]]

# CHECK: ^[[BODY]]:
# CHECK: py.getGlobalHandle @x
# ... rest of if
# CHECK: br ^[[TEST]]

# CHECK: br ^[[THEN]]

# CHECK: ^[[THEN]]:
