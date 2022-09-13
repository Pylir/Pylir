# RUN: pylir %s -emit-pylir -o - -c -S | FileCheck %s

# CHECK-LABEL: __init__

global x

x()

# CHECK: %[[X_LOADED:.*]] = py.load @x
# CHECK: %[[IS_UNBOUND:.*]] = py.isUnboundValue %[[X_LOADED]]
# CHECK: cond_br %[[IS_UNBOUND]], ^{{[[:alnum:]]+}}, ^[[HAPPY_PATH:[[:alnum:]]+]]

# CHECK: ^[[HAPPY_PATH]]:
# CHECK-DAG: %[[TUPLE:.*]] = py.makeTuple ()
# CHECK-DAG: %[[DICT:.*]] = py.constant(#py.dict<{}>)
# CHECK: py.call @pylir__call__(%[[X_LOADED]], %[[TUPLE]], %[[DICT]])

x(5, k=3)

# CHECK: %[[X_LOADED:.*]] = py.load @x
# CHECK: %[[IS_UNBOUND:.*]] = py.isUnboundValue %[[X_LOADED]]
# CHECK: cond_br %[[IS_UNBOUND]], ^{{[[:alnum:]]+}}, ^[[HAPPY_PATH:[[:alnum:]]+]]

# CHECK: ^[[HAPPY_PATH]]:
# CHECK: %[[FIVE:.*]] = py.constant(#py.int<5>)
# CHECK: %[[NAME:.*]] = py.constant(#py.str<"k">)
# CHECK: %[[HASH:.*]] = py.str.hash %[[NAME]]
# CHECK: %[[THREE:.*]] = py.constant(#py.int<3>)
# CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[FIVE]])
# CHECK: %[[DICT:.*]] = py.makeDict (%[[NAME]] hash(%[[HASH]]) : %[[THREE]])
# CHECK: py.call @pylir__call__(%[[X_LOADED]], %[[TUPLE]], %[[DICT]])

x(*(), **{})

# CHECK: %[[X_LOADED:.*]] = py.load @x
# CHECK: %[[IS_UNBOUND:.*]] = py.isUnboundValue %[[X_LOADED]]
# CHECK: cond_br %[[IS_UNBOUND]], ^{{[[:alnum:]]+}}, ^[[HAPPY_PATH:[[:alnum:]]+]]

# CHECK: ^[[HAPPY_PATH]]:
# CHECK: %[[TUPLE1:.*]] = py.makeTuple ()
# CHECK: %[[DICT1:.*]] = py.makeDict ()
# CHECK: %[[TUPLE:.*]] = py.makeTuple (*%[[TUPLE1]])
# CHECK: %[[DICT:.*]] = py.makeDict (**%[[DICT1]])
# CHECK: py.call @pylir__call__(%[[X_LOADED]], %[[TUPLE]], %[[DICT]])
