# RUN: pylir %s -emit-pylir -o - -c -S | FileCheck %s

# CHECK-LABEL: @"bin_ops$impl[0]"
# CHECK-SAME: %{{[[:alnum:]]+}}
# CHECK-SAME: %[[A:[[:alnum:]]+]]
# CHECK-SAME: %[[B:[[:alnum:]]+]]

def bin_ops(a, b):
    a + b
    # CHECK: call @pylir__add__(%[[A]], %[[B]])

    a - b
    # CHECK: call @pylir__sub__(%[[A]], %[[B]])

    a | b
    # CHECK: call @pylir__or__(%[[A]], %[[B]])

    a ^ b
    # CHECK: call @pylir__xor__(%[[A]], %[[B]])

    a & b
    # CHECK: call @pylir__and__(%[[A]], %[[B]])

    a << b
    # CHECK: call @pylir__lshift__(%[[A]], %[[B]])

    a >> b
    # CHECK: call @pylir__rshift__(%[[A]], %[[B]])

    a * b
    # CHECK: call @pylir__mul__(%[[A]], %[[B]])

    a / b
    # CHECK: call @pylir__div__(%[[A]], %[[B]])

    a // b
    # CHECK: call @pylir__floordiv__(%[[A]], %[[B]])

    a @ b
    # CHECK: call @pylir__matmul__(%[[A]], %[[B]])

    a in b
    # CHECK: call @pylir__contains__(%[[B]], %[[A]])


# CHECK-LABEL: @"unary_ops$impl[0]"
# CHECK-SAME: %{{[[:alnum:]]+}}
# CHECK-SAME: %[[A:[[:alnum:]]+]]

def unary_ops(a):
    -a
    # CHECK: call @pylir__neg__(%[[A]])
    +a
    # CHECK: call @pylir__pos__(%[[A]])
    ~a
    # CHECK: call @pylir__invert__(%[[A]])


# CHECK-LABEL: @"boolean_ops$impl[0]"
# CHECK-SAME: %{{[[:alnum:]]+}}
# CHECK-SAME: %[[A:[[:alnum:]]+]]
# CHECK-SAME: %[[B:[[:alnum:]]+]]
def boolean_ops(a, b):
    global c
    # Testing with calls on purpose to check in the output that side effects
    # are correctly sequenced

    c = a() and b()
    # CHECK: %[[TUPLE:.*]] = makeTuple ()
    # CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
    # CHECK: %[[RES:.*]] = call @pylir__call__(%[[A]], %[[TUPLE]], %[[DICT]])
    # CHECK: %[[BOOL:.*]] = constant(#py.ref<@builtins.bool>)
    # CHECK: %[[TYPE:.*]] = typeOf %[[RES]]
    # CHECK: %[[IS_BOOL:.*]] = is %[[TYPE]], %[[BOOL]]
    # CHECK: cf.cond_br %[[IS_BOOL]], ^[[CONTINUE:.*]](%[[RES]] : !py.dynamic), ^[[CALC_BOOL:[[:alnum:]]+]]
    # CHECK: ^[[CALC_BOOL]]:
    # CHECK: %[[TUPLE:.*]] = makeTuple (%[[RES]])
    # CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
    # CHECK: %[[RES_AS_BOOL:.*]] = call @pylir__call__(%[[BOOL]], %[[TUPLE]], %[[DICT]])
    # CHECK: cf.br ^[[CONTINUE]](%[[RES_AS_BOOL]] : !py.dynamic)
    # CHECK: ^[[CONTINUE]](%[[RES_AS_BOOL:.*]]: !py.dynamic loc({{.*}})):
    # CHECK: %[[RES_AS_I1:.*]] = bool_toI1 %[[RES_AS_BOOL]]
    # CHECK: cf.cond_br %[[RES_AS_I1]], ^[[CALCULATE_B_BLOCK:.*]], ^[[DEST:.*]](%[[RES]] : !py.dynamic)

    # CHECK: ^[[CALCULATE_B_BLOCK]]:
    # CHECK: %[[TUPLE:.*]] = makeTuple ()
    # CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
    # CHECK: %[[RES:.*]] = call @pylir__call__(%[[B]], %[[TUPLE]], %[[DICT]])
    # CHECK: cf.br ^[[DEST]](%[[RES]] : !py.dynamic)

    # CHECK: ^[[DEST]](
    # CHECK-SAME: %[[ARG:[[:alnum:]]+]]
    # CHECK: store %[[ARG]] : !py.dynamic into @c

    c = a() or b()
    # CHECK: %[[TUPLE:.*]] = makeTuple ()
    # CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
    # CHECK: %[[RES:.*]] = call @pylir__call__(%[[A]], %[[TUPLE]], %[[DICT]])
    # CHECK: %[[BOOL:.*]] = constant(#py.ref<@builtins.bool>)
    # CHECK: %[[TYPE:.*]] = typeOf %[[RES]]
    # CHECK: %[[IS_BOOL:.*]] = is %[[TYPE]], %[[BOOL]]
    # CHECK: cf.cond_br %[[IS_BOOL]], ^[[CONTINUE:.*]](%[[RES]] : !py.dynamic), ^[[CALC_BOOL:[[:alnum:]]+]]
    # CHECK: ^[[CALC_BOOL]]:
    # CHECK: %[[TUPLE:.*]] = makeTuple (%[[RES]])
    # CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
    # CHECK: %[[RES_AS_BOOL:.*]] = call @pylir__call__(%[[BOOL]], %[[TUPLE]], %[[DICT]])
    # CHECK: cf.br ^[[CONTINUE]](%[[RES_AS_BOOL]] : !py.dynamic)
    # CHECK: ^[[CONTINUE]](%[[RES_AS_BOOL:.*]]: !py.dynamic loc({{.*}})):
    # CHECK: %[[RES_AS_I1:.*]] = bool_toI1 %[[RES_AS_BOOL]]
    # CHECK: cf.cond_br %[[RES_AS_I1]], ^[[DEST:.*]](%[[RES]] : !py.dynamic), ^[[CALCULATE_B_BLOCK:[[:alnum:]]+]]

    # CHECK: ^[[CALCULATE_B_BLOCK]]:
    # CHECK: %[[TUPLE:.*]] = makeTuple ()
    # CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
    # CHECK: %[[RES:.*]] = call @pylir__call__(%[[B]], %[[TUPLE]], %[[DICT]])
    # CHECK: cf.br ^[[DEST]](%[[RES]] : !py.dynamic)

    # CHECK: ^[[DEST]](
    # CHECK-SAME: %[[ARG:[[:alnum:]]+]]
    # CHECK: store %[[ARG]] : !py.dynamic into @c

    c = not a
    # CHECK: %[[BOOL:.*]] = constant(#py.ref<@builtins.bool>)
    # CHECK: %[[TYPE:.*]] = typeOf %[[A]]
    # CHECK: %[[IS_BOOL:.*]] = is %[[TYPE]], %[[BOOL]]
    # CHECK: cf.cond_br %[[IS_BOOL]], ^[[CONTINUE:.*]](%[[A]] : !py.dynamic), ^[[CALC_BOOL:[[:alnum:]]+]]
    # CHECK: ^[[CALC_BOOL]]:
    # CHECK: %[[TUPLE:.*]] = makeTuple (%[[A]])
    # CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
    # CHECK: %[[RES_AS_BOOL:.*]] = call @pylir__call__(%[[BOOL]], %[[TUPLE]], %[[DICT]])
    # CHECK: cf.br ^[[CONTINUE]](%[[RES_AS_BOOL]] : !py.dynamic)
    # CHECK: ^[[CONTINUE]](%[[RES_AS_BOOL:.*]]: !py.dynamic loc({{.*}})):
    # CHECK: %[[RES_AS_I1:.*]] = bool_toI1 %[[RES_AS_BOOL]]
    # CHECK: %[[TRUE:.*]] = arith.constant true
    # CHECK: %[[INVERTED:.*]] = arith.xori %[[TRUE]], %[[RES_AS_I1]]
    # CHECK: %[[AS_BOOL:.*]] = bool_fromI1 %[[INVERTED]]
    # CHECK: store %[[AS_BOOL]] : !py.dynamic into @c


# CHECK-LABEL: @"aug_assign_ops$impl[0]"
# CHECK-SAME: %{{[[:alnum:]]+}}
# CHECK-SAME: %[[A:[[:alnum:]]+]]
# CHECK-SAME: %[[B:[[:alnum:]]+]]
def aug_assign_ops(a, b):
    a += b
    # CHECK: %[[A_1:.*]] = call @pylir__iadd__(%[[A]], %[[B]])
    a -= b
    # CHECK: %[[A_2:.*]] = call @pylir__isub__(%[[A_1]], %[[B]])
    a *= b
    # CHECK: %[[A_3:.*]] = call @pylir__imul__(%[[A_2]], %[[B]])
    a /= b
    # CHECK: %[[A_4:.*]] = call @pylir__idiv__(%[[A_3]], %[[B]])
    a //= b
    # CHECK: %[[A_5:.*]] = call @pylir__ifloordiv__(%[[A_4]], %[[B]])
    a %= b
    # CHECK: %[[A_6:.*]] = call @pylir__imod__(%[[A_5]], %[[B]])
    a @= b
    # CHECK: %[[A_7:.*]] = call @pylir__imatmul__(%[[A_6]], %[[B]])
    a &= b
    # CHECK: %[[A_8:.*]] = call @pylir__iand__(%[[A_7]], %[[B]])
    a |= b
    # CHECK: %[[A_9:.*]] = call @pylir__ior__(%[[A_8]], %[[B]])
    a ^= b
    # CHECK: %[[A_10:.*]] = call @pylir__ixor__(%[[A_9]], %[[B]])
    a >>= b
    # CHECK: %[[A_11:.*]] = call @pylir__irshift__(%[[A_10]], %[[B]])
    a <<= b
    # CHECK: %[[A_12:.*]] = call @pylir__ilshift__(%[[A_11]], %[[B]])
    return a
    # CHECK: return %[[A_12]]
