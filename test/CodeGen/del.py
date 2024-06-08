# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK-LABEL: init "__main__"

# CHECK: %[[STR:.*]] = py.constant(#py.str<"x">)
# CHECK: %[[HASH:.*]] = py.str_hash %[[STR]]
# CHECK: %[[X:.*]] = py.dict_tryGetItem %{{.*}}[%[[STR]] hash(%[[HASH]])]
# CHECK: %[[UNBOUND:.*]] = py.isUnboundValue %[[X]]
# CHECK: cf.cond_br %[[UNBOUND]], ^[[RAISE:.*]], ^[[CONT:[[:alnum:]]+]]

# CHECK: ^[[RAISE]]:
# CHECK: %[[TYPE:.*]] = py.constant
# CHECK: %[[EXC:.*]] = call %[[TYPE]]()
# CHECK: py.raise %[[EXC]]

# CHECK: ^[[CONT]]:
# CHECK: %[[UNBOUND:.*]] = py.constant(#py.unbound)
# CHECK: %[[STR:.*]] = py.constant(#py.str<"x">)
# CHECK: %[[HASH:.*]] = py.str_hash %[[STR]]
# CHECK: py.dict_setItem %{{.*}}[%[[STR]] hash(%[[HASH]])] to %[[UNBOUND]]
del x


# CHECK-LABEL: class "__main__.Test"
class Test:
    # CHECK-NEXT: %[[DICT:[[:alnum:]]+]]
    # CHECK: py.dict_delItem %{{.*}} hash(%{{.*}}) from %[[DICT]]
    del foo


# CHECK-LABEL: func "__main__.bar"
def bar():
    # Once for the check whether foobar exists
    # CHECK: %[[UNBOUND:.*]] = py.constant(#py.unbound)
    # CHECK: py.isUnboundValue %[[UNBOUND]]
    del foobar

    # Another for the use of foobar.
    # CHECK: %[[UNBOUND:.*]] = py.constant(#py.unbound)
    # CHECK: py.isUnboundValue %[[UNBOUND]]
    return foobar


# CHECK: %[[UNBOUND:.*]] = py.constant(#py.unbound)
# CHECK: %[[STR:.*]] = py.constant(#py.str<"a">)
# CHECK: %[[HASH:.*]] = py.str_hash %[[STR]]
# CHECK: py.dict_setItem %{{.*}}[%[[STR]] hash(%[[HASH]])] to %[[UNBOUND]]
# CHECK: %[[UNBOUND:.*]] = py.constant(#py.unbound)
# CHECK: %[[STR:.*]] = py.constant(#py.str<"b">)
# CHECK: %[[HASH:.*]] = py.str_hash %[[STR]]
# CHECK: py.dict_setItem %{{.*}}[%[[STR]] hash(%[[HASH]])] to %[[UNBOUND]]
del a, b
# CHECK: %[[UNBOUND:.*]] = py.constant(#py.unbound)
# CHECK: %[[STR:.*]] = py.constant(#py.str<"x">)
# CHECK: %[[HASH:.*]] = py.str_hash %[[STR]]
# CHECK: py.dict_setItem %{{.*}}[%[[STR]] hash(%[[HASH]])] to %[[UNBOUND]]
# CHECK: %[[UNBOUND:.*]] = py.constant(#py.unbound)
# CHECK: %[[STR:.*]] = py.constant(#py.str<"y">)
# CHECK: %[[HASH:.*]] = py.str_hash %[[STR]]
# CHECK: py.dict_setItem %{{.*}}[%[[STR]] hash(%[[HASH]])] to %[[UNBOUND]]
del [x, y]

# CHECK: %[[STR:.*]] = py.constant(#py.str<"bar">)
# CHECK: %[[HASH:.*]] = py.str_hash %[[STR]]
# CHECK: %[[BAR:.*]] = py.dict_tryGetItem %{{.*}}[%[[STR]] hash(%[[HASH]])]
# CHECK: %[[INDEX:.*]] = py.constant(#py.int<0>)
# CHECK: delItem %[[BAR]][%[[INDEX]]]
del bar[0]
