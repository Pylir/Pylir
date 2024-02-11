# RUN: pylir %s -Xnew-codegen -emit-pylir -o - -c -S | FileCheck %s

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
