# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

def foo():
    raise TypeError

# CHECK-LABEL: func private @"foo$impl[0]"
# CHECK: %[[TYPE_ERROR:.*]] = py.constant @builtins.TypeError
# CHECK: %[[TYPE_OF:.*]] = py.typeOf %[[TYPE_ERROR]]
# subclass check...
# CHECK: cond_br %{{.*}}, ^[[TYPE_BLOCK:.*]], ^[[INSTANCE_BLOCK:.*]](%[[TYPE_ERROR]]

# CHECK: ^[[TYPE_BLOCK]]:
# CHECK: %[[BASE_EXCEPTION:.*]] = py.constant @builtins.BaseException
# subclass check... + type error

# CHECK: %[[TUPLE:.*]] = py.constant #py.tuple<()>
# CHECK: %[[DICT:.*]] = py.constant #py.dict<{}>
# ... call
# CHECK: br ^[[INSTANCE_BLOCK]]
# CHECK-SAME: %{{[[:alnum:]]+}}

# CHECK: ^[[INSTANCE_BLOCK]](
# CHECK-SAME: %[[INSTANCE:[[:alnum:]]+]]
# CHECK: %[[TYPE_OF:.*]] = py.typeOf %[[INSTANCE]]
# CHECK: %[[BASE_EXCEPTION:.*]] = py.constant @builtins.BaseException
# subclass check + type error

# CHECK: py.raise %[[INSTANCE]]
