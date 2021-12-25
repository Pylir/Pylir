# RUN: pylir %s -emit-pylir -o - -c -S | FileCheck %s

x = 0


def outer():
    class Foo:
        if False:
            x = 3
        y = x

# CHECK-LABEL: outer

# CHECK-DAG: %[[BASES:.*]] = py.constant #py.tuple<()>
# CHECK-DAG: %[[KEYWORDS:.*]] = py.constant #py.dict<{}>
# CHECK-DAG: %[[NAME:.*]] = py.constant #py.str<"outer.<locals>.Foo">
# CHECK: py.makeClass %[[NAME]], @[[FUNC:.*]], %[[BASES]], %[[KEYWORDS]]

# CHECK: func private @[[FUNC]]
# CHECK-SAME: %[[CELL:[[:alnum:]]+]]: !py.dynamic
# CHECK-SAME: %[[DICT:[[:alnum:]]+]]: !py.dynamic

# CHECK: %[[COND:.*]] = py.bool.toI1
# CHECK: cond_br %[[COND]], ^[[TRUE:.*]], ^[[FALSE:[[:alnum:]]+]]

# CHECK: ^[[TRUE]]:
# CHECK: %[[THREE:.*]] = py.constant #py.int<3>
# CHECK: %[[X:.*]] = py.constant #py.str<"x">
# CHECK: py.dict.setItem %[[DICT]][%[[X]]] to %[[THREE]]

# CHECK: ^[[FALSE]]:
# CHECK: %[[X:.*]] = py.constant #py.str<"x">
# CHECK: %[[ITEM:.*]], %[[FOUND:.*]] = py.dict.tryGetItem %[[DICT]][%[[X]]]
# CHECK: cond_br %[[FOUND]], ^[[DICT_FOUND:.*]](%[[ITEM]] : !py.dynamic), ^[[ELSE:[[:alnum:]]+]]

# CHECK: ^[[ELSE]]:
# CHECK: py.load @x
# ...
# CHECK: br ^[[DICT_FOUND]]

# CHECK: ^[[DICT_FOUND]](%[[RESULT:[[:alnum:]]+]]: !py.dynamic {{.*}}):
# CHECK: %[[Y:.*]] = py.constant #py.str<"y">
# CHECK: py.dict.setItem %[[DICT]][%[[Y]]] to %[[RESULT]]

# CHECK: return %[[DICT]]
