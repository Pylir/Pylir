# RUN: pylir %s -emit-pylir -o - -c -S | FileCheck %s

# XFAIL: *

# CHECK-LABEL: __init__

global x

x()

# CHECK: %[[X_LOADED:.*]] = py.load @x
# CHECK: %[[IS_UNBOUND:.*]] = py.isUnboundValue %[[X_LOADED]]
# CHECK: cond_br %[[IS_UNBOUND]], ^{{[[:alnum:]]+}}, ^[[HAPPY_PATH:[[:alnum:]]+]]

# CHECK: ^[[HAPPY_PATH]]:
# CHECK-DAG: %[[TUPLE:.*]] = py.constant #py.tuple<()>
# CHECK-DAG: %[[DICT:.*]] = py.constant #py.dict<{}>
# CHECK: %[[RESULT:.*]], %[[SUCCESS:.*]] = py.getFunction %[[X_LOADED]]
# CHECK: cond_br %[[SUCCESS]], ^[[HAPPY_PATH:[[:alnum:]]+]]

# CHECK: ^[[HAPPY_PATH]]:
# CHECK: %[[FUNCTION:.*]] = py.function.getFunction %[[RESULT]]
# CHECK: call_indirect %[[FUNCTION]](%[[RESULT]], %[[TUPLE]], %[[DICT]])

x(5, k=3)

# CHECK: %[[X_LOADED:.*]] = py.load @x
# CHECK: %[[IS_UNBOUND:.*]] = py.isUnboundValue %[[X_LOADED]]
# CHECK: cond_br %[[IS_UNBOUND]], ^{{[[:alnum:]]+}}, ^[[HAPPY_PATH:[[:alnum:]]+]]

# CHECK: ^[[HAPPY_PATH]]:
# CHECK: %[[FIVE:.*]] = py.constant #py.int<5>
# CHECK: %[[K:.*]] = py.constant #py.str<"k">
# CHECK: %[[THREE:.*]] = py.constant #py.int<3>
# CHECK-DAG: %[[TUPLE:.*]] = py.makeTuple (%[[FIVE]])
# CHECK-DAG: %[[DICT:.*]] = py.makeDict (%[[K]] : %[[THREE]])
# CHECK: %[[RESULT:.*]], %[[SUCCESS:.*]] = py.getFunction %[[X_LOADED]]
# CHECK: cond_br %[[SUCCESS]], ^[[HAPPY_PATH:[[:alnum:]]+]]

# CHECK: ^[[HAPPY_PATH]]:
# CHECK: %[[FUNCTION:.*]] = py.function.getFunction %[[RESULT]]
# CHECK: call_indirect %[[FUNCTION]](%[[RESULT]], %[[TUPLE]], %[[DICT]])

x(*(), **{})

# CHECK: %[[X_LOADED:.*]] = py.load @x
# CHECK: %[[IS_UNBOUND:.*]] = py.isUnboundValue %[[X_LOADED]]
# CHECK: cond_br %[[IS_UNBOUND]], ^{{[[:alnum:]]+}}, ^[[HAPPY_PATH:[[:alnum:]]+]]

# CHECK: ^[[HAPPY_PATH]]:
# CHECK: %[[ARG1:.*]] = py.constant #py.tuple<()>
# CHECK: %[[ARG2:.*]] = py.makeDict ()
# CHECK-DAG: %[[TUPLE:.*]] = py.makeTuple (*%[[ARG1]])
# CHECK-DAG: %[[DICT:.*]] = py.makeDict (**%[[ARG2]])
# CHECK: %[[RESULT:.*]], %[[SUCCESS:.*]] = py.getFunction %[[X_LOADED]]
# CHECK: cond_br %[[SUCCESS]], ^[[HAPPY_PATH:[[:alnum:]]+]]

# CHECK: ^[[HAPPY_PATH]]:
# CHECK: %[[FUNCTION:.*]] = py.function.getFunction %[[RESULT]]
# CHECK: call_indirect %[[FUNCTION]](%[[RESULT]], %[[TUPLE]], %[[DICT]])
