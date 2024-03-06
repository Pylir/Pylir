# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK-LABEL: init "__main__"
# CHECK-DAG: %[[THREE:.*]] = py.constant(#py.int<3>)
# CHECK-DAG: %[[ONE:.*]] = py.constant(#py.int<1>)
# CHECK: func "__main__.<lambda>"(%[[ARG0:.*]] "a" = %[[THREE]], %[[ARG1:.*]] only "c" = %[[ONE]]) {
# CHECK: %[[RET:.*]] = binOp %[[ARG0]] __add__ %[[ARG1]]
# CHECK: return %[[RET]]
l = lambda a=3, *, c=1: a + c
