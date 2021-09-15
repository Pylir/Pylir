# RUN: pylir %s -emit-mlir -o - | FileCheck %s

# CHECK-LABEL: __init__

global x

x()

# CHECK-DAG: %[[TUPLE:.*]] = py.makeTuple ()
# CHECK-DAG: %[[DICT:.*]] = py.makeDict ()
# CHECK: py.call %{{[[:alnum:]]+}}(* %[[TUPLE]], * * %[[DICT]])

x(5, k=3)

# CHECK: %[[FIVE:.*]] = py.constant #py.int<5>
# CHECK: %[[K:.*]] = py.constant "k"
# CHECK: %[[THREE:.*]] = py.constant #py.int<3>
# CHECK-DAG: %[[TUPLE:.*]] = py.makeTuple (%[[FIVE]])
# CHECK-DAG: %[[DICT:.*]] = py.makeDict (%[[K]] : %[[THREE]])
# CHECK: py.call %{{[[:alnum:]]+}}(* %[[TUPLE]], * * %[[DICT]])

x(*(), **{})

# CHECK: %[[ARG1:.*]] = py.makeTuple ()
# CHECK: %[[ARG2:.*]] = py.makeDict ()
# CHECK-DAG: %[[TUPLE:.*]] = py.makeTuple (*%[[ARG1]])
# CHECK-DAG: %[[DICT:.*]] = py.makeDict (**%[[ARG2]])
# CHECK: py.call %{{[[:alnum:]]+}}(* %[[TUPLE]], * * %[[DICT]])
