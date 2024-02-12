# RUN: pylir %s -Xnew-codegen -emit-pylir -o - -c -S | FileCheck %s

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
    # CHECK: %[[I1:.*]] = py.bool_toI1 %[[TO_BOOL]]
    # CHECK: cf.cond_br %[[I1]], ^[[BB1:.*]], ^[[BB2:.*]](%[[I1]] : i1)

    # CHECK: ^[[BB1]]:
    # CHECK: %[[B_CALL:.*]] = call %[[B]]()
    # CHECK: %[[BOOL:.*]] = py.constant(#[[$BOOL]])
    # CHECK: %[[TO_BOOL:.*]] = call %[[BOOL]](%[[B_CALL]])
    # CHECK: %[[I1:.*]] = py.bool_toI1 %[[TO_BOOL]]
    # CHECK: cf.br ^[[BB2]](%[[I1]] : i1)

    # CHECK: ^[[BB2]](%[[RES:.*]]: i1 loc({{.*}})):
    # CHECK: py.bool_fromI1 %[[RES]]
    c = a() and b()

    # CHECK: %[[A_CALL:.*]] = call %[[A]]()
    # CHECK: %[[BOOL:.*]] = py.constant(#[[$BOOL]])
    # CHECK: %[[TO_BOOL:.*]] = call %[[BOOL]](%[[A_CALL]])
    # CHECK: %[[I1:.*]] = py.bool_toI1 %[[TO_BOOL]]
    # CHECK: cf.cond_br %[[I1]], ^[[BB4:.*]](%[[I1]] : i1), ^[[BB3:[[:alnum:]]+]]

    # CHECK: ^[[BB3]]:
    # CHECK: %[[B_CALL:.*]] = call %[[B]]()
    # CHECK: %[[BOOL:.*]] = py.constant(#[[$BOOL]])
    # CHECK: %[[TO_BOOL:.*]] = call %[[BOOL]](%[[B_CALL]])
    # CHECK: %[[I1:.*]] = py.bool_toI1 %[[TO_BOOL]]
    # CHECK: cf.br ^[[BB4]](%[[I1]] : i1)

    # CHECK: ^[[BB4]](%[[RES:.*]]: i1 loc({{.*}})):
    # CHECK: py.bool_fromI1 %[[RES]]
    c = a() or b()

    # CHECK: %[[A_CALL:.*]] = call %[[A]]()
    # CHECK: %[[BOOL:.*]] = py.constant(#[[$BOOL]])
    # CHECK: %[[TO_BOOL:.*]] = call %[[BOOL]](%[[A_CALL]])
    # CHECK: %[[I1:.*]] = py.bool_toI1 %[[TO_BOOL]]
    # CHECK: %[[TRUE:.*]] = arith.constant true
    # CHECK: %[[NOT:.*]] = arith.xori %[[I1]], %[[TRUE]]
    # CHECK: py.bool_fromI1 %[[NOT]]
    c = not a()
