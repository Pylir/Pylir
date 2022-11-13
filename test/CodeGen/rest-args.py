# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

def foo(*args, **kwd):
    pass


# CHECK-LABEL: func private @"foo$cc[0]"
# CHECK-SAME: %[[SELF:[[:alnum:]]+]]
# CHECK-SAME: %[[TUPLE:[[:alnum:]]+]]
# CHECK-SAME: %[[DICT:[[:alnum:]]+]]
# CHECK: %[[ZERO:.*]] = arith.constant 0
# CHECK: %[[ARGS:.*]] = py.tuple.dropFront %[[ZERO]], %[[TUPLE]]
# CHECK: call @"foo$impl[0]"(%[[SELF]], %[[ARGS]], %[[DICT]])

def bar(a, *args, k, **kwd):
    pass

# CHECK-LABEL: func private @"bar$cc[0]"
# CHECK-SAME: %[[SELF:[[:alnum:]]+]]
# CHECK-SAME: %[[TUPLE:[[:alnum:]]+]]
# CHECK-SAME: %[[DICT:[[:alnum:]]+]]

# CHECK: %[[TUPLE_LEN:.*]] = py.tuple.len %[[TUPLE]]

# ... processing of a

# processing of *args
# CHECK: %[[TUPLE_ARG:.*]] = py.tuple.dropFront %{{.*}}, %[[TUPLE]]

# processing of k...
# CHECK: %[[CONSTANT:.*]] = py.constant(#py.str<"k">)
# CHECK: %[[CONSTANT_HASH:.*]] = py.str.hash %[[CONSTANT]]
# CHECK: py.dict.delItem %[[CONSTANT]] hash(%[[CONSTANT_HASH]]) from %[[DICT]]

# CHECK: call @"bar$impl[0]"(%[[SELF]], %[[BAR_A:[[:alnum:]]+]], %[[TUPLE_ARG]], %[[BAR_K:[[:alnum:]]+]], %[[DICT]])
