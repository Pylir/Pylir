# RUN: pylir %s -emit-pylir -o - -c -S | FileCheck %s

# CHECK: #[[$BOOL:.*]] = #py.globalValue<builtins.bool{{>|,}}

# CHECK-LABEL: func "__main__.binary_ops"
# CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
# CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
def binary_ops(a, b):
    # CHECK: binOp %[[ARG0]] __lt__ %[[ARG1]]
    a < b

    # CHECK: binOp %[[ARG0]] __le__ %[[ARG1]]
    a <= b

    # CHECK: binOp %[[ARG0]] __gt__ %[[ARG1]]
    a > b

    # CHECK: binOp %[[ARG0]] __ge__ %[[ARG1]]
    a >= b

    # CHECK: binOp %[[ARG0]] __eq__ %[[ARG1]]
    a == b

    # CHECK: binOp %[[ARG0]] __ne__ %[[ARG1]]
    a != b

    # CHECK: %[[IS:.*]] = py.is %[[ARG0]], %[[ARG1]]
    # CHECK: py.bool_fromI1 %[[IS]]
    a is b

    # CHECK: contains %[[ARG0]] in %[[ARG1]]
    a in b


# CHECK-LABEL: func "__main__.chaining"
# CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
# CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
# CHECK-SAME: %[[ARG2:[[:alnum:]]+]]
def chaining(a, b, c):
    # CHECK: %[[CARG0:.*]] = call %[[ARG0]]
    # CHECK: %[[CARG1:.*]] = call %[[ARG1]]
    # CHECK: %[[BIN1:.*]] = binOp %[[CARG0]] __lt__ %[[CARG1]]
    # CHECK: %[[BOOL:.*]] = py.constant(#[[$BOOL]])
    # CHECK: %[[CALL:.*]] = call %[[BOOL]](%[[BIN1]])
    # CHECK: cf.br ^[[IS_BOOL_BB:.*]](%[[CALL]] : !py.dynamic)
    # CHECK: ^[[IS_BOOL_BB]](%[[BOOL:.*]]: !py.dynamic loc({{.*}})):
    # CHECK: %[[I1:.*]] = py.bool_toI1 %[[BOOL]]
    # CHECK: cf.cond_br %[[I1]], ^[[BB5:.*]], ^[[BB8:.*]](%[[BIN1]] : !py.dynamic)

    # CHECK: ^[[BB5]]:
    # CHECK: %[[CARG2:.*]] = call %[[ARG2]]
    # CHECK: %[[BIN2:.*]] = binOp %[[CARG1]] __lt__ %[[CARG2]]
    # CHECK: cf.br ^[[BB8]](%[[BIN2]] : !py.dynamic)

    # CHECK: ^[[BB8]](%[[VALUE:.*]]: !py.dynamic loc({{.*}})):
    # CHECK: return %[[VALUE]]
    return a() < b() < c()


# CHECK-LABEL: func "__main__.invert"
# CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
# CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
def invert(a, b):
    # CHECK: %[[IS:.*]] = py.is %[[ARG0]], %[[ARG1]]
    # CHECK: %[[B:.*]] = py.bool_fromI1 %[[IS]]
    # CHECK: %[[BOOL:.*]] = py.constant(#[[$BOOL]])
    # CHECK: %[[CALL:.*]] = call %[[BOOL]](%[[B]])
    # CHECK: cf.br ^[[IS_BOOL_BB:.*]](%[[CALL]] : !py.dynamic)
    # CHECK: ^[[IS_BOOL_BB]](%[[BOOL:.*]]: !py.dynamic loc({{.*}})):
    # CHECK: %[[I1:.*]] = py.bool_toI1 %[[BOOL]]
    # CHECK: %[[TRUE:.*]] = arith.constant true
    # CHECK: %[[INV:.*]] = arith.xori %[[I1]], %[[TRUE]]
    # CHECK: %[[B:.*]] = py.bool_fromI1 %[[INV]]
    # CHECK: return %[[B]]
    return a is not b
