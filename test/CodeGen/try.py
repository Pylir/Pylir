# RUN: pylir %s -emit-pylir -o - | FileCheck %s

def foo():
    try:
        pass
    except 0:
        pass


# CHECK-LABEL: func private @"foo$impl[0]"
# CHECK-NOT: %{{.*}} =
# CHECK: %[[VALUE:.*]] =
# CHECK-NEXT: return %[[VALUE]]

def bar(param):
    try:
        param()
    except TypeError:
        return 1
    except:
        return 0

# CHECK-LABEL: func private @"bar$impl[0]"
# CHECK: %[[TYPE_ERROR:.*]] = py.constant @builtins.TypeError
# CHECK: %[[NEW_METHOD:.*]] = py.getSlot "__new__" from %[[TYPE_ERROR]]
# CHECK: %[[NEW_METHOD_PTR:.*]] = py.function.getFunction %[[NEW_METHOD]]
# CHECK: %[[EXCEPTION:.*]] = call_indirect %[[NEW_METHOD_PTR]]
# CHECK: br ^[[EXCEPTION_HANDLER:[[:alnum:]]+]](
# CHECK-SAME: %[[EXCEPTION]]

# CHECK: ^[[EXCEPTION_HANDLER]](
# CHECK-SAME: %[[EXCEPTION:[[:alnum:]]+]]

# CHECK: %[[MATCHING:.*]] = py.constant @builtins.TypeError
# CHECK: %[[TUPLE_TYPE:.*]] = py.constant @builtins.tuple
# CHECK: %[[MATCHING_TYPE:.*]] = py.typeOf %[[MATCHING]]
# CHECK: %[[IS_TUPLE:.*]] = py.is %[[MATCHING_TYPE]], %[[TUPLE_TYPE]]
# CHECK: cond_br %[[IS_TUPLE]], ^[[TUPLE_BLOCK:.*]], ^[[EXCEPTION_BLOCK:[[:alnum:]]+]]

# CHECK: ^[[EXCEPTION_BLOCK]]:
# subclass of BaseException check...
# CHECK: %[[EXCEPTION_TYPE:.*]] = py.typeOf %[[EXCEPTION]]
# CHECK: %[[MRO:.*]] = py.getSlot "__mro__" from %[[EXCEPTION_TYPE]]
# CHECK: %[[IS_SUBCLASS:.*]] = py.linearContains %[[MATCHING]] in %[[MRO]]
# CHECK: cond_br %[[IS_SUBCLASS]], ^[[SUITE_BLOCK:.*]], ^[[SKIP_BLOCK:[[:alnum:]]+]]

# skipping tuple case for now

# CHECK: ^[[SUITE_BLOCK]]:
# CHECK: %[[ONE:.*]] = py.constant #py.int<1>
# CHECK: return %[[ONE]]

# CHECK: ^[[SKIP_BLOCK]]:
# CHECK: %[[ZERO:.*]] = py.constant #py.int<0>
# CHECK: return %[[ZERO]]
