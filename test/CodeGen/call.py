# RUN: pylir %s -emit-mlir -o - | FileCheck %s

# CHECK-LABEL: __init__

global x

x()

# CHECK: %[[X_LOADED:.*]] = py.load @x
# CHECK: %[[TUPLE:.*]] = py.makeTuple ()
# CHECK: %[[DICT:.*]] = py.constant #py.dict<{}>
# CHECK: call_indirect %{{.*}}(%{{.*}}, %[[TUPLE]], %[[DICT]])

x(5, k=3)

# CHECK: %[[X_LOADED:.*]] = py.load @x
# CHECK: %[[FIVE:.*]] = py.constant #py.int<5>
# CHECK: %[[K:.*]] = py.constant "k"
# CHECK: %[[THREE:.*]] = py.constant #py.int<3>
# CHECK-DAG: %[[TUPLE:.*]] = py.makeTuple (%[[FIVE]])
# CHECK-DAG: %[[DICT:.*]] = py.makeDict (%[[K]] : %[[THREE]])
# ... mro lookup
# CHECK: call_indirect %{{.*}}(%{{.*}}, %[[TUPLE]], %[[DICT]])

x(*(), **{})

# CHECK: %[[X_LOADED:.*]] = py.load @x
# CHECK: %[[ARG1:.*]] = py.makeTuple ()
# CHECK: %[[ARG2:.*]] = py.makeDict ()
# CHECK-DAG: %[[TUPLE:.*]] = py.makeTuple (*%[[ARG1]])
# CHECK-DAG: %[[DICT:.*]] = py.makeDict (**%[[ARG2]])
# ... mro lookup
# CHECK: call_indirect %{{.*}}(%{{.*}}, %[[TUPLE]], %[[DICT]])
