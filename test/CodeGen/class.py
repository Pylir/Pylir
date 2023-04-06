# RUN: pylir %s -emit-pylir -o - -c -S | FileCheck %s

# XFAIL: *

x = 0


def outer():
    class Foo:
        if False:
            x = 3
        y = x

# CHECK-LABEL: outer

# CHECK-DAG: %[[BASES:.*]] = constant #py.tuple<()>
# CHECK-DAG: %[[KEYWORDS:.*]] = constant #py.dict<{}>
# CHECK-DAG: %[[NAME:.*]] = constant #py.str<"outer.<locals>.Foo">
# CHECK: makeClass %[[NAME]], @[[FUNC:.*]], %[[BASES]], %[[KEYWORDS]]

# CHECK: func private @[[FUNC]]
# CHECK-SAME: %[[CELL:[[:alnum:]]+]]: !py.dynamic
# CHECK-SAME: %[[DICT:[[:alnum:]]+]]: !py.dynamic

# CHECK: %[[COND:.*]] = bool_toI1
# CHECK: cond_br %[[COND]], ^[[TRUE:.*]], ^[[FALSE:[[:alnum:]]+]]

# CHECK: ^[[TRUE]]:
# CHECK: %[[THREE:.*]] = constant #py.int<3>
# CHECK: %[[X:.*]] = constant #py.str<"x">
# CHECK: dict_setItem %[[DICT]][%[[X]]] to %[[THREE]]

# CHECK: ^[[FALSE]]:
# CHECK: %[[X:.*]] = constant #py.str<"x">
# CHECK: %[[ITEM:.*]], %[[FOUND:.*]] = dict_tryGetItem %[[DICT]][%[[X]]]
# CHECK: cond_br %[[FOUND]], ^[[DICT_FOUND:.*]](%[[ITEM]] : !py.dynamic), ^[[ELSE:[[:alnum:]]+]]

# CHECK: ^[[ELSE]]:
# CHECK: load @x
# ...
# CHECK: br ^[[DICT_FOUND]]

# CHECK: ^[[DICT_FOUND]](%[[RESULT:[[:alnum:]]+]]: !py.dynamic {{.*}}):
# CHECK: %[[Y:.*]] = constant #py.str<"y">
# CHECK: dict_setItem %[[DICT]][%[[Y]]] to %[[RESULT]]

# CHECK: return %[[DICT]]
