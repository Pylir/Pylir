# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

def foo(*args, **kwd):
    pass


# CHECK-LABEL: func private @"foo$cc[0]"
# CHECK-SAME: %[[SELF:[[:alnum:]]+]]
# CHECK-SAME: %[[TUPLE:[[:alnum:]]+]]
# CHECK-SAME: %[[DICT:[[:alnum:]]+]]

# CHECK: call @"foo$impl[0]"(%[[SELF]], %[[TUPLE]], %[[DICT]])

def bar(a, *args, k, **kwd):
    pass

# CHECK-LABEL: func private @"bar$cc[0]"
# CHECK-SAME: %[[SELF:[[:alnum:]]+]]
# CHECK-SAME: %[[TUPLE:[[:alnum:]]+]]
# CHECK-SAME: %[[DICT:[[:alnum:]]+]]

# CHECK: %[[TUPLE_LEN:.*]] = py.tuple.len %[[TUPLE]]

# ... processing of a

# list created first
# CHECK: %[[LIST:.*]] = py.makeList ()
# CHECK: %[[START:.*]] = arith.constant 1
# CHECK: br ^[[CONDITION:[[:alnum:]]+]]
# CHEK-SAME: %[[START]]

# CHECK: ^[[CONDITION]]
# CHECK-SAME: %[[ITERATOR:[[:alnum:]]+]]

# CHECK: %[[CHECK:.*]] = arith.cmpi ult, %[[ITERATOR]], %[[TUPLE_LEN]]
# CHECK: cond_br %[[CHECK]], ^[[BODY:[[:alnum:]]+]], ^[[END:[[:alnum:]]+]]

# CHECK: ^[[BODY]]:
# CHECK: %[[FETCHED:.*]] = py.tuple.getItem %[[TUPLE]][
# CHECK-SAME: %[[ITERATOR]]
# CHECK: py.list.append %[[LIST]], %[[FETCHED]]
# CHECK: %[[ONE:.*]] = arith.constant 1
# CHECK: %[[INCREMENTED:.*]] = arith.addi %[[ITERATOR]], %[[ONE]]
# CHECK: br ^[[CONDITION]]
# CHECK-SAME: %[[INCREMENTED]]

# CHECK: ^[[END]]:
# CHECK: %[[TUPLE_ARG:.*]] = py.list.toTuple %[[LIST]]

# processing of k...
# CHECK: %[[CONSTANT:.*]] = py.constant(#py.str<value = "k">)
# CHECK: py.dict.delItem %[[CONSTANT]] from %[[DICT]]

# CHECK: call @"bar$impl[0]"(%[[SELF]], %[[BAR_A:[[:alnum:]]+]], %[[TUPLE_ARG]], %[[BAR_K:[[:alnum:]]+]], %[[DICT]])
