# RUN: pylir %s -emit-pylir -o - -c -S | FileCheck %s

# CHECK-LABEL: func @__init__

global x

x()

# CHECK: %[[X_LOADED:.*]] = load @x
# CHECK: %[[IS_UNBOUND:.*]] = isUnboundValue %[[X_LOADED]]
# CHECK: cond_br %[[IS_UNBOUND]], ^{{[[:alnum:]]+}}, ^[[HAPPY_PATH:[[:alnum:]]+]]

# CHECK: ^[[HAPPY_PATH]]:
# CHECK-DAG: %[[TUPLE:.*]] = makeTuple ()
# CHECK-DAG: %[[DICT:.*]] = constant(#py.dict<{}>)
# CHECK: call @pylir__call__(%[[X_LOADED]], %[[TUPLE]], %[[DICT]])

x(5, k=3)

# CHECK: %[[X_LOADED:.*]] = load @x
# CHECK: %[[IS_UNBOUND:.*]] = isUnboundValue %[[X_LOADED]]
# CHECK: cond_br %[[IS_UNBOUND]], ^{{[[:alnum:]]+}}, ^[[HAPPY_PATH:[[:alnum:]]+]]

# CHECK: ^[[HAPPY_PATH]]:
# CHECK: %[[FIVE:.*]] = constant(#py.int<5>)
# CHECK: %[[NAME:.*]] = constant(#py.str<"k">)
# CHECK: %[[HASH:.*]] = str_hash %[[NAME]]
# CHECK: %[[THREE:.*]] = constant(#py.int<3>)
# CHECK: %[[TUPLE:.*]] = makeTuple (%[[FIVE]])
# CHECK: %[[DICT:.*]] = makeDict (%[[NAME]] hash(%[[HASH]]) : %[[THREE]])
# CHECK: call @pylir__call__(%[[X_LOADED]], %[[TUPLE]], %[[DICT]])

x(*(), **{})

# CHECK: %[[X_LOADED:.*]] = load @x
# CHECK: %[[IS_UNBOUND:.*]] = isUnboundValue %[[X_LOADED]]
# CHECK: cond_br %[[IS_UNBOUND]], ^{{[[:alnum:]]+}}, ^[[HAPPY_PATH:[[:alnum:]]+]]

# CHECK: ^[[HAPPY_PATH]]:
# CHECK: %[[TUPLE1:.*]] = makeTuple ()
# CHECK: %[[DICT1:.*]] = makeDict ()
# CHECK: %[[TUPLE:.*]] = makeTuple (*%[[TUPLE1]])
# CHECK: %[[DICT:.*]] = makeDict (**%[[DICT1]])
# CHECK: call @pylir__call__(%[[X_LOADED]], %[[TUPLE]], %[[DICT]])
