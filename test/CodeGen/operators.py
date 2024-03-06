# RUN: pylir %s -emit-pylir -o - -c -S | FileCheck %s

# CHECK: #[[$BOOL:.*]] = #py.globalValue<builtins.bool{{,|>}}

# CHECK-LABEL: func "__main__.bin_ops"
# CHECK-SAME: %[[A:[[:alnum:]]+]]
# CHECK-SAME: %[[B:[[:alnum:]]+]]
def bin_ops(a, b):
    # CHECK: binOp %[[A]] __add__ %[[B]]
    a + b

    # CHECK: binOp %[[A]] __sub__ %[[B]]
    a - b

    # CHECK: binOp %[[A]] __or__ %[[B]]
    a | b

    # CHECK: binOp %[[A]] __xor__ %[[B]]
    a ^ b

    # CHECK: binOp %[[A]] __and__ %[[B]]
    a & b

    # CHECK: binOp %[[A]] __lshift__ %[[B]]
    a << b

    # CHECK: binOp %[[A]] __rshift__ %[[B]]
    a >> b

    # CHECK: binOp %[[A]] __mul__ %[[B]]
    a * b

    # CHECK: binOp %[[A]] __div__ %[[B]]
    a / b

    # CHECK: binOp %[[A]] __floordiv__ %[[B]]
    a // b

    # CHECK: binOp %[[A]] __matmul__ %[[B]]
    a @ b


# CHECK-LABEL: func "__main__.boolean_ops"
# CHECK-SAME: %[[A:[[:alnum:]]+]]
# CHECK-SAME: %[[B:[[:alnum:]]+]]
def boolean_ops(a, b):
    # CHECK: %[[A_CALL:.*]] = call %[[A]]()
    # CHECK: %[[BOOL:.*]] = py.constant(#[[$BOOL]])
    # CHECK: %[[TO_BOOL:.*]] = call %[[BOOL]](%[[A_CALL]])
    # CHECK: cf.br ^[[IS_BOOL_BB:.*]](%[[TO_BOOL]] : !py.dynamic)
    # CHECK: ^[[IS_BOOL_BB]](%[[BOOL:.*]]: !py.dynamic loc({{.*}})):
    # CHECK: %[[I1:.*]] = py.bool_toI1 %[[BOOL]]
    # CHECK: cf.cond_br %[[I1]], ^[[BB1:.*]], ^[[BB2:.*]](%[[I1]] : i1)

    # CHECK: ^[[BB1]]:
    # CHECK: %[[B_CALL:.*]] = call %[[B]]()
    # CHECK: %[[BOOL:.*]] = py.constant(#[[$BOOL]])
    # CHECK: %[[TO_BOOL:.*]] = call %[[BOOL]](%[[B_CALL]])
    # CHECK: cf.br ^[[IS_BOOL_BB:.*]](%[[TO_BOOL]] : !py.dynamic)
    # CHECK: ^[[IS_BOOL_BB]](%[[BOOL:.*]]: !py.dynamic loc({{.*}})):
    # CHECK: %[[I1:.*]] = py.bool_toI1 %[[BOOL]]
    # CHECK: cf.br ^[[BB2]](%[[I1]] : i1)

    # CHECK: ^[[BB2]](%[[RES:.*]]: i1 loc({{.*}})):
    # CHECK: py.bool_fromI1 %[[RES]]
    c = a() and b()

    # CHECK: %[[A_CALL:.*]] = call %[[A]]()
    # CHECK: %[[BOOL:.*]] = py.constant(#[[$BOOL]])
    # CHECK: %[[TO_BOOL:.*]] = call %[[BOOL]](%[[A_CALL]])
    # CHECK: cf.br ^[[IS_BOOL_BB:.*]](%[[TO_BOOL]] : !py.dynamic)
    # CHECK: ^[[IS_BOOL_BB]](%[[BOOL:.*]]: !py.dynamic loc({{.*}})):
    # CHECK: %[[I1:.*]] = py.bool_toI1 %[[BOOL]]
    # CHECK: cf.cond_br %[[I1]], ^[[BB4:.*]](%[[I1]] : i1), ^[[BB3:[[:alnum:]]+]]

    # CHECK: ^[[BB3]]:
    # CHECK: %[[B_CALL:.*]] = call %[[B]]()
    # CHECK: %[[BOOL:.*]] = py.constant(#[[$BOOL]])
    # CHECK: %[[TO_BOOL:.*]] = call %[[BOOL]](%[[B_CALL]])
    # CHECK: cf.br ^[[IS_BOOL_BB:.*]](%[[TO_BOOL]] : !py.dynamic)
    # CHECK: ^[[IS_BOOL_BB]](%[[BOOL:.*]]: !py.dynamic loc({{.*}})):
    # CHECK: %[[I1:.*]] = py.bool_toI1 %[[BOOL]]
    # CHECK: cf.br ^[[BB4]](%[[I1]] : i1)

    # CHECK: ^[[BB4]](%[[RES:.*]]: i1 loc({{.*}})):
    # CHECK: py.bool_fromI1 %[[RES]]
    c = a() or b()

    # CHECK: %[[A_CALL:.*]] = call %[[A]]()
    # CHECK: %[[BOOL:.*]] = py.constant(#[[$BOOL]])
    # CHECK: %[[TO_BOOL:.*]] = call %[[BOOL]](%[[A_CALL]])
    # CHECK: cf.br ^[[IS_BOOL_BB:.*]](%[[TO_BOOL]] : !py.dynamic)
    # CHECK: ^[[IS_BOOL_BB]](%[[BOOL:.*]]: !py.dynamic loc({{.*}})):
    # CHECK: %[[I1:.*]] = py.bool_toI1 %[[BOOL]]
    # CHECK: %[[TRUE:.*]] = arith.constant true
    # CHECK: %[[NOT:.*]] = arith.xori %[[I1]], %[[TRUE]]
    # CHECK: py.bool_fromI1 %[[NOT]]
    c = not a()


# CHECK-LABEL: func "__main__.assign_ops"
# CHECK-SAME: %[[A:[[:alnum:]]+]]
# CHECK-SAME: %[[B:[[:alnum:]]+]]
def assign_ops(a, b):
    # CHECK: %[[A2:.*]] = binAssignOp %[[A]] __iadd__ %[[B]]
    a += b

    # CHECK: %[[A3:.*]] = binAssignOp %[[A2]] __isub__ %[[B]]
    a -= b

    # CHECK: %[[A4:.*]] = binAssignOp %[[A3]] __ior__ %[[B]]
    a |= b

    # CHECK: %[[A5:.*]] = binAssignOp %[[A4]] __ixor__ %[[B]]
    a ^= b

    # CHECK: %[[A6:.*]] = binAssignOp %[[A5]] __iand__ %[[B]]
    a &= b

    # CHECK: %[[A7:.*]] = binAssignOp %[[A6]] __ilshift__ %[[B]]
    a <<= b

    # CHECK: %[[A8:.*]] = binAssignOp %[[A7]] __irshift__ %[[B]]
    a >>= b

    # CHECK: %[[A9:.*]] = binAssignOp %[[A8]] __imul__ %[[B]]
    a *= b

    # CHECK: %[[A10:.*]] = binAssignOp %[[A9]] __idiv__ %[[B]]
    a /= b

    # CHECK: %[[A11:.*]] = binAssignOp %[[A10]] __ifloordiv__ %[[B]]
    a //= b

    # CHECK: binAssignOp %[[A11]] __imatmul__ %[[B]]
    a @= b
